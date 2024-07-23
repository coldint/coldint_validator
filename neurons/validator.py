# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import copy
import datetime as dt
import functools
import os
import json
import math
import pickle
import time
import torch
import random
import asyncio
import numpy as np
import requests

import wandb
import constants
import dataset
import validation
from model import model_utils
from model.data import ModelId
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from neurons import config
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console

import bittensor as bt
from utilities.miner_iterator import MinerIterator
from utilities import utils
from utilities.perf_monitor import PerfMonitor

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Container:
    '''Empty container object'''
    pass

class Validator:
    TRACKER_FILENAME = "model_tracker_2.pickle"
    UIDS_FILENAME = "uids_2.pickle"
    VERSION_FILENAME = "version.txt"
    BENCHMARK_FILENAME = "benchmark.json"

    def state_path(self) -> str:
        """
        Returns the file path for storing validator state.

        Returns:
        str: A string representing the file path.
        """
        return os.path.join(self.config.model_dir, "vali-state")

    def load_state(self):
        # Construct the filepaths to save/load state.
        state_dir = self.state_path()
        os.makedirs(state_dir, exist_ok=True)

        self.uids_filepath = os.path.join(state_dir, Validator.UIDS_FILENAME)
        self.tracker_filepath = os.path.join(state_dir, Validator.TRACKER_FILENAME)
        self.version_filepath = os.path.join(state_dir, Validator.VERSION_FILENAME)

        # Check if the version has changed since we last restarted.
        previous_version = utils.get_version(self.version_filepath)
        utils.save_version(self.version_filepath, constants.__spec_version__)

        # If this is an upgrade, blow away state so that everything is re-evaluated.
        if previous_version != constants.__spec_version__:
            bt.logging.info(
                f"Validator updated. Previous version={previous_version}. Current version={constants.__spec_version__}"
            )
            if os.path.exists(self.uids_filepath):
                bt.logging.info(
                    f"Because the validator updated, deleting {self.uids_filepath} so everything is re-evaluated."
                )
                os.remove(self.uids_filepath)
            if os.path.exists(self.tracker_filepath):
                bt.logging.info(
                    f"Because the validator updated, deleting {self.tracker_filepath} so everything is re-evaluated."
                )
                os.remove(self.tracker_filepath)

        # Initialize the model tracker.
        if not os.path.exists(self.tracker_filepath):
            bt.logging.warning("No tracker state file found. Starting from scratch.")
        else:
            try:
                self.model_tracker.load_state(self.tracker_filepath)
            except Exception as e:
                bt.logging.warning(
                    f"Failed to load model tracker state. Reason: {e}. Starting from scratch."
                )

        # Initialize the UIDs to eval.
        if not os.path.exists(self.uids_filepath):
            bt.logging.warning("No uids state file found. Starting from scratch.")
        else:
            try:
                with open(self.uids_filepath, "rb") as f:
                    self.uids_to_eval = pickle.load(f)
                    self.pending_uids_to_eval = pickle.load(f)
            except Exception as e:
                bt.logging.warning(
                    f"Failed to load uids to eval state. Reason: {e}. Starting from scratch."
                )
                # We also need to wipe the tracker state in this case to ensure we re-evaluate all the models.
                self.model_tracker = ModelTracker()
                if os.path.exists(self.tracker_filepath):
                    bt.logging.warning(
                        f"Because the uids to eval state failed to load, deleting tracker state at {self.tracker_filepath} so everything is re-evaluated."
                    )
                    os.remove(self.tracker_filepath)

    def __init__(self):
        self.config = config.validator_config()
        bt.logging(config=self.config)
        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid, lite=False)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # Dont log to wandb if offline.
        self.wandb_run = None
        if self.config.wandb.on and not self.config.offline:
            self.new_wandb_run()

        # === Running args ===
        self.weights = torch.zeros(constants.SUBNET_N_UIDS)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        self.uids_to_eval = set()

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.pending_uids_to_eval_lock = threading.RLock()
        self.pending_uids_to_eval = set()

        # Setup a model tracker to track which miner is using which model id.
        self.model_tracker = ModelTracker()

        self.hall_of_fame = {}

        self.load_state()

        # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        self.metadata_store = ChainModelMetadataStore(
            self.subtensor, self.wallet, self.config.netuid
        )

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        bt.logging.trace("Starting ModelUpdater")
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

        # Create a metagraph lock to avoid cross thread access issues in the update and clean loop.
        self.metagraph_lock = threading.RLock()

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        bt.logging.trace("Starting update_models thread")
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

        # == Initialize the cleaner thread to remove outdated models ==
        bt.logging.trace("Starting clean_models thread")
        self.clean_thread = threading.Thread(target=self.clean_models, daemon=True)
        self.clean_thread.start()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()
            self.clean_thread.join()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""

        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project=constants.WANDB_PROJECT,
            entity=constants.WANDB_ENTITY,
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__validator_version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def save_state(self):
        """Saves the state of the validator to a file."""

        bt.logging.trace("Saving validator state.")
        if not os.path.exists(self.state_path()):
            os.makedirs(self.state_path())

        with self.pending_uids_to_eval_lock:
            # Save the state of the validator uids to file.
            with open(self.uids_filepath, "wb") as f:
                pickle.dump(self.uids_to_eval, f)
                pickle.dump(self.pending_uids_to_eval, f)

        # Save the state of the tracker to file.
        self.model_tracker.save_state(self.tracker_filepath)

    def check_top_models(self):
        # At most once per `chain_update_cadence`, check which models are being assigned weight by
        # the top validators and ensure they'll be evaluated soon.
        now = dt.datetime.now()
        if self.last_checked_top_models_time is not None and \
                (now - self.last_checked_top_models_time) < constants.chain_update_cadence:
            return

        self.last_checked_top_models_time = now
        with self.metagraph_lock:
            metagraph = copy.deepcopy(self.metagraph)

        # Find any miner UIDs which top valis are assigning weight and aren't currently scheduled for an eval.
        top_miner_uids = set(utils.list_top_miners(metagraph))
        bt.logging.info(f"Top miners: {top_miner_uids}")
        with self.pending_uids_to_eval_lock:
            uids_to_add = (
                top_miner_uids
                - self.uids_to_eval
                - self.pending_uids_to_eval
            )

        for uid in uids_to_add:
            # Limit how often we'll retry these top models.
            time_diff = (
                now - self.uid_last_retried_evaluation[uid]
                if uid in self.uid_last_retried_evaluation
                else constants.model_retry_cadence  # Default to being stale enough to check again.
            )
            if time_diff < constants.model_retry_cadence:
                continue

            try:
                self.uid_last_retried_evaluation[uid] = now

                # Redownload this model and schedule it for eval.
                hotkey = metagraph.hotkeys[uid]
                asyncio.run(
                    self.model_updater.sync_model(hotkey, force=True)
                )

                # Since this is a top model (as determined by other valis),
                # we don't worry if self.pending_uids is already "full". At most
                # there can be 10 top models that we'd add here and that would be
                # a wildy exceptional case. It would require every vali to have a
                # different top model.
                self.pending_uids_to_eval.add(uid)
                bt.logging.debug(
                    f"Retrying evaluation for previously discarded model with incentive for UID={uid}."
                )
            except Exception:
                bt.logging.debug(
                    f"Failure in update loop for UID={uid} during top model check. {traceback.format_exc()}"
                )

    def update_hall_of_fame(self):
        now = time.time()
        if self.last_hof_fetch is not None and \
                (now - self.last_hof_fetch) < constants.HOF_FETCH_INTERVAL:
            return
        self.last_hof_fetch = now

        try:
            req = requests.get(constants.HOF_URL)
            req.raise_for_status()
            self.hall_of_fame = req.json()
            bt.logging.info(f"Fetched hall of fame content, containing {len(self.hall_of_fame)} entries")
        except Exception as e:
            bt.logging.error(f"Failed to fetch hall of fame: {e}")


    def update_models(self):
        """
        Updates the models in the local store based on the latest metadata from the chain.
        Periodically fetch hall of fame config json.
        """

        # Track how recently we updated each uid from sequential iteration.
        uid_last_checked_sequential = dict()
        # Track how recently we checked the list of top models.
        self.last_checked_top_models_time = None
        # Track how recently we retried a model with incentive we've already dropped.
        self.uid_last_retried_evaluation = dict()
        # Track when we last fetched hall of fame
        self.last_hof_fetch = None

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                self.update_hall_of_fame()
                self.check_top_models()

                # Top model check complete. Now continue with the sequential iterator to check for the next miner
                # to update.
                pending_uid_count = 0
                current_uid_count = 0
                with self.pending_uids_to_eval_lock:
                    pending_uid_count = len(self.pending_uids_to_eval)
                    current_uid_count = len(self.uids_to_eval)

                # Only allow at most sample max models. Typically this will be carryover from sample_min + new models.
                while pending_uid_count + current_uid_count >= self.config.sample_max:
                    # Wait 5 minutes for the eval loop to process them.
                    bt.logging.info(
                        f"Update loop: Already {pending_uid_count + current_uid_count} synced models pending eval. Checking again in 5 minutes."
                    )
                    time.sleep(300)
                    # Check to see if the pending uids have been cleared yet.
                    with self.pending_uids_to_eval_lock:
                        pending_uid_count = len(self.pending_uids_to_eval)
                        current_uid_count = len(self.uids_to_eval)

                # We have space to add more models for eval. Process the next UID.
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't already checked it in the chain update cadence.
                time_diff = (
                    dt.datetime.now() - uid_last_checked_sequential[next_uid]
                    if next_uid in uid_last_checked_sequential
                    else None
                )
                if time_diff and time_diff < constants.chain_update_cadence:
                    # If we have seen it within chain update cadence then sleep until it has been at least that long.
                    time_to_sleep = (
                        constants.chain_update_cadence - time_diff
                    ).total_seconds()
                    bt.logging.trace(
                        f"Update loop has already processed all UIDs in the last {constants.chain_update_cadence}. Sleeping {time_to_sleep} seconds."
                    )
                    time.sleep(time_to_sleep)

                uid_last_checked_sequential[next_uid] = dt.datetime.now()

                # Get their hotkey from the metagraph.
                with self.metagraph_lock:
                    hotkey = self.metagraph.hotkeys[next_uid]

                # Sync the model, if necessary.
                updated = asyncio.run(
                    self.model_updater.sync_model(hotkey, force=False)
                )
                if updated:
                    with self.pending_uids_to_eval_lock:
                        self.pending_uids_to_eval.add(next_uid)
                        bt.logging.debug(
                            f"Found a new model for UID={next_uid}. It will be evaluated on the next loop."
                        )
            except Exception as e:
                bt.logging.error(
                    f"Error in update loop: {e} \n {traceback.format_exc()}"
                )

        bt.logging.info("Exiting update models loop.")

    def clean_models(self):
        """Cleans up models that are no longer referenced."""

        # Delay the clean-up thread until the update loop has had time to run one full pass after an upgrade.
        # This helps prevent unnecessarily deleting a model which is on disk, but hasn't yet been re-added to the
        # model tracker by the update loop.
        time.sleep(dt.timedelta(hours=1).total_seconds())

        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                bt.logging.trace("Starting cleanup of stale models.")

                # Get a mapping of all hotkeys to model ids.
                hotkey_to_model_metadata = (
                    self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
                )
                hotkey_to_model_id = {
                    hotkey: metadata.id
                    for hotkey, metadata in hotkey_to_model_metadata.items()
                }

                # Find all hotkeys that are currently being evaluated or pending eval.
                uids_to_keep = set()
                with self.pending_uids_to_eval_lock:
                    uids_to_keep = self.uids_to_eval.union(self.pending_uids_to_eval)

                hotkeys_to_keep = set()
                with self.metagraph_lock:
                    for uid in uids_to_keep:
                        hotkeys_to_keep.add(self.metagraph.hotkeys[uid])

                # Only keep those hotkeys.
                evaluated_hotkeys_to_model_id = {
                    hotkey: model_id
                    for hotkey, model_id in hotkey_to_model_id.items()
                    if hotkey in hotkeys_to_keep
                }

                self.local_store.delete_unreferenced_models(
                    valid_models_by_hotkey=evaluated_hotkeys_to_model_id,
                    grace_period_seconds=300,
                )
            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}")

            # Only check every 5 minutes.
            time.sleep(dt.timedelta(minutes=5).total_seconds())

        bt.logging.info("Exiting clean models loop.")

    async def try_set_weights(self, ttl: int):
        """Sets the weights on the chain with ttl, without raising exceptions if it times out."""

        async def _try_set_weights():
            try:
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.weights[:len(self.metagraph.uids)],
                    wait_for_inclusion=False,
                    version_key=constants.weights_version_key,
                )
            except:
                bt.logging.warning("Failed to set weights. Trying again later.")

            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All non-zero weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                if weight == 0:
                    continue
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

        try:
            bt.logging.debug(f"Setting weights.")
            await asyncio.wait_for(_try_set_weights(), ttl)
            bt.logging.debug(f"Finished setting weights.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")

    async def try_sync_metagraph(self, ttl: int):
        """Syncs the metagraph with ttl in a background process, without raising exceptions if it times out."""

        def sync_metagraph(endpoint):
            metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid, lite=False)
            metagraph.save()

        process = multiprocessing.Process(
            target=sync_metagraph, args=(self.subtensor.chain_endpoint,)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return

        bt.logging.info("Synced metagraph")
        with self.metagraph_lock:
            self.metagraph.load()
            self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())
            self.model_tracker.on_hotkeys_updated(set(self.metagraph.hotkeys))

    async def try_run_step(self, ttl: int):
        """Runs a step with ttl in a background process, without raising exceptions if it times out."""

        async def _try_run_step():
            await self.run_step()

        try:
            bt.logging.trace("Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.trace("Finished running step.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    def get_reward_weights(self):
        reward_weights = torch.zeros_like(self.weights)
        with self.metagraph_lock:
            metagraph = copy.deepcopy(self.metagraph)
        current_block = metagraph.block.item()

        for uid, hotkey in enumerate(metagraph.hotkeys):
            if hotkey not in self.hall_of_fame:
                continue
            try:
                for entry in self.hall_of_fame[hotkey]:
                    delta_blocks = current_block - entry['block']
                    delta_epochs = delta_blocks / constants.blocks_per_epoch
                    iv = entry['reward'] / constants.REWARDS_IV_FACTOR
                    reward_weights[uid] += iv * constants.REWARDS_DECAY_FACTOR**delta_epochs
                    bt.logging.info(f"Rewarding UID {uid} / {hotkey} with weight {reward_weights[uid]:.04f} for '{entry.get('desc','')}'")
            except Exception as e:
                bt.logging.warning(f"Reward computation for UID {uid} / {hotkey} failed ({e})")

        # Make sure total reward weight does not exceed BOUNTIES_MAX_FRACTION
        total_weight = reward_weights.sum().item()
        if total_weight > constants.REWARDS_MAX_FRACTION:
            reward_weights *= constants.REWARDS_MAX_FRACTION / total_weight
            total_weight = constants.REWARDS_MAX_FRACTION

        return reward_weights, total_weight

    def update_weights(self, uids, model_weights):
        '''
        Update self.weights, based on uids and model_weights.
        Up to constants.REWARDS_MAX_FRACTION part of the total weight will be based
        on the hall of fame rewards settings.
        '''
        new_weights = torch.zeros_like(self.weights)
        reward_weights, reward_sum = self.get_reward_weights()
        # Scale model weights down by (1 - reward_sum)
        for uid, weight in model_weights.items():
            new_weights[uid] = (1 - reward_sum) * weight
        # Add bounties
        new_weights += reward_weights
        # Normalize total, which should in principle already be the case
        new_weights /= new_weights.sum()

        # First time around, use weights without EMA
        if self.weights.count_nonzero().item() == 0:
            self.weights = new_weights
        else:
            self.weights = constants.weight_alpha * self.weights + (1 - constants.weight_alpha) * new_weights
        self.weights = self.weights.nan_to_num(0.0)

    def load_benchmark_config(self):
        if not os.path.exists(self.BENCHMARK_FILENAME):
            return {}
        try:
            with open(self.BENCHMARK_FILENAME) as f:
                d = {int(k): v for k, v in json.load(f).items()}
            return d
        except Exception as e:
            bt.logging.warning(f"Failed to load benchmark config: {e}")
        return {}

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
            1. Identifies valid models for evaluation (top N from last run + newly updated models).
            2. Generates random pages for evaluation and prepares batches for each page from the dataset.
            3. Computes the scoring for each model based on the losses incurred on the evaluation batches.
            4. Calculates wins and win rates for each model to determine their performance relative to others.
            5. Updates the weights of each model based on their performance and applies a softmax normalization.
            6. Implements a blacklist mechanism to remove underperforming models from the evaluation set.
            7. Logs all relevant data for the step, including model IDs, pages, batches, wins, win rates, and losses.
        """
        # Add uids with newly updated models to the upcoming batch of evaluations.
        with self.pending_uids_to_eval_lock:
            self.uids_to_eval.update(self.pending_uids_to_eval)
            self.pending_uids_to_eval.clear()

        benchmark_cfg = self.load_benchmark_config()
        uids = list(self.uids_to_eval | set(benchmark_cfg.keys()))

        if not uids:
            bt.logging.debug("No uids to eval. Waiting 5 minutes to download some models.")
            time.sleep(300)
            return

        uid_to_block = {uid: math.inf for uid in uids}
        bt.logging.debug(f'run_step() @ current block {self.current_block}')

        tokenizer = model_utils.get_tokenizer(cache_dir=self.config.model_dir)
        dataloader = dataset.SubsetFineWebEdu2Loader(
            batch_size=constants.batch_size,
            num_pages=self.config.pages_per_eval,
            tokenizer=tokenizer,
            pack=False
        )
        batches = list(dataloader)

        bt.logging.debug(f"Computing losses on {uids} with {len(batches)} batches from pages {dataloader.pages}")

        if len(batches) == 0:
            bt.logging.warning(f"No samples to eval. Waiting one minute before retrying.")
            time.sleep(60)
            return

        # Compute model losses on batches.
        losses_per_uid = {muid: None for muid in uids}

        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

        for uid in uids:
            bt.logging.trace(f"Computing model losses for uid {uid}.")

            if uid in benchmark_cfg:
                # Model data from dynamic config
                bcfg = benchmark_cfg[uid]
                hotkey = bcfg.get("hotkey", "xxx")
                model_path = bcfg.get('path', 'please specify path')
                model_i_metadata = Container()
                model_i_metadata.block = bcfg.get("block", 1<<31)
                model_i_metadata.id = ModelId.dummy(bcfg.get('identifier',model_path))
            elif uid < len(self.metagraph.hotkeys):
                # Model from chain
                hotkey = self.metagraph.hotkeys[uid]
                model_i_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                model_path = None
            else:
                model_i_metadata = None

            # This variable should be overwritten below if the model has metadata.
            losses = [math.inf for _ in range(len(batches))]

            if model_i_metadata != None:
                try:
                    uid_to_block[uid] = model_i_metadata.block if model_i_metadata.block is not None else 1<<31
                    bt.logging.debug(f"Evaluating uid {uid} from block {uid_to_block[uid]}")

                    with load_model_perf.sample():
                        model_i = self.local_store.retrieve_model(hotkey, model_i_metadata.id, path=model_path)

                    with compute_loss_perf.sample():
                        losses = utils.run_in_subprocess(
                            functools.partial(
                                validation.compute_losses,
                                model_i.pt_model,
                                batches,
                                self.config.device,
                                tokenizer.eos_token_id,
                            ),
                            ttl=360,
                            mode="spawn",
                        )
                    del model_i
                except Exception as e:
                    bt.logging.error(
                        f"Error in eval loop: {e}. Setting losses for uid: {uid} to infinity.\n{traceback.format_exc()}"
                    )
            else:
                bt.logging.debug(
                    f"Unable to load metadata for {uid}. Setting loss to infinity."
                )

            losses_per_uid[uid] = losses
            bt.logging.debug(f"Losses for uid:{uid}: {np.nanmean(losses):.03f} +- {np.nanstd(losses):.03f}")

        win_info = validation.compute_wins(losses_per_uid, uid_to_block, self.current_block)
        if 'win_rate' not in win_info:
            bt.logging.warning("compute_wins() returned no result")
            return

        # Skew weight distribution towards models with high win_rate
        win_rate = np.array([win_info['win_rate'][uid] for uid in uids])
        model_weights = win_rate**constants.WEIGHT_SKEW_FACTOR
        model_weights /= np.sum(model_weights)
        model_weights = {uid: weight for uid, weight in zip(uids, model_weights)}
        self.update_weights(uids, model_weights)

        # Model sort order: by weight if >= 0.001, win-rate otherwise
        model_prio = {
            uid: (
                # Add 1 to ensure it is always greater than a win rate.
                1 + self.weights[uid].item()
                if self.weights[uid].item() >= 0.001
                else wr
            )
            for uid, wr in win_info['win_rate'].items()
        }
        self.uids_to_eval = set(sorted(model_prio, key=model_prio.get, reverse=True)[:self.config.sample_min])

        self.save_state()

        # Log the performance of the eval loop.
        bt.logging.debug(load_model_perf.summary_str())
        bt.logging.debug(compute_loss_perf.summary_str())

        # Log to screen and wandb.
        pages_desc = [f'{cfg_name}_{num_rows}_{split}' for cfg_name, num_rows, split in dataloader.pages]
        self.log_step(
            uids,
            uid_to_block,
            pages_desc,
            win_info,
            benchmark_cfg,
            losses_per_uid,
            load_model_perf.summary_str(),
            compute_loss_perf.summary_str(),
        )

        # Increment the number of completed run steps by 1
        self.run_step_count += 1

    def print_win_matrix(self, matrix, benchmark_cfg=None, show_delta_loss=False):
        if show_delta_loss:
            title = "Model win matrix, true wins/adv wins/avg delta loss"
        else:
            title = "Model win matrix, true wins/adv wins"
        table = Table(title=title)
        table.add_column("win \ lose", justify="right", style="cyan", no_wrap=True)
        for uid in matrix:
            table.add_column(f'UID {uid}')
        for uid_a,row in matrix.items():
            label = ''
            if uid_a in benchmark_cfg:
                label = benchmark_cfg[uid_a].get('label', benchmark_cfg[uid_a].get('path', '???')) + ' '
            vals = [f'{label}UID {uid_a}']
            for uid_b,wins in row.items():
                val = '?'
                if uid_a==uid_b:
                    val = '...'
                elif wins is not None:
                    if show_delta_loss:
                        val = f'{wins["wins"]}/{wins["wins_adv"]}/{wins["loss"]:.01f}'
                    else:
                        val = f'{wins["wins"]}/{wins["wins_adv"]}'
                vals.append(val)
            table.add_row(*vals)
        console = Console()
        console.print(table)

    def log_step(
        self,
        uids,
        uid_to_block,
        pages,
        win_info,
        benchmark_cfg,
        losses_per_uid,
        load_model_perf_str,
        compute_loss_perf_str,
    ):
        """Logs the results of the step to the console and wandb (if enabled)."""
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "pages": pages,
            "uids": uids,
            "uid_data": {},
        }
        wins = win_info.get('wins', {})
        win_rate = win_info.get('win_rate', {})
        advantage_factors = win_info.get('advantage_factors', {})
        for i, uid in enumerate(uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": uid_to_block[uid],
                "loss_avg": np.nanmean(losses_per_uid[uid]),
                "loss_std": np.nanstd(losses_per_uid[uid]),
                "adv_factor": 100*(1-advantage_factors.get(uid,1)),
                "win_rate": win_rate.get(uid, 0),
                "win_total": wins.get(uid, 0),
                "weight": self.weights[uid].item(),
            }
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_loss", style="magenta")
        table.add_column("adv_factor(%)", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in uids:
            d = step_log["uid_data"][str(uid)]
            try:
                table.add_row(
                    str(uid),
                    f"{d['loss_avg']:.04f}+-{d['loss_std']:.04f}",
                    f"{d['adv_factor']:.03f} of {100*constants.advantage_initial:.03f}",
                    str(round(d["win_rate"], 4)),
                    str(d["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(d["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        win_matrix = win_info.get('matrix', None)
        if win_matrix is not None:
            self.print_win_matrix(win_matrix, benchmark_cfg)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        bt.logging.trace(f"Step results: {step_log}")

        if self.config.wandb.on and not self.config.offline:
            # If we have already completed X steps then we will complete the current wandb run and make a new one.
            if (
                self.run_step_count
                and self.run_step_count % constants.MAX_RUN_STEPS_PER_WANDB_RUN == 0
            ):
                bt.logging.trace(
                    f"Validator has completed {self.run_step_count} run steps. Creating a new wandb run."
                )
                self.wandb_run.finish()
                self.new_wandb_run()

            original_format_json = json.dumps(step_log)
            uids = step_log["uids"]
            uid_data = step_log["uid_data"]

            # Create a new dictionary with the required format
            graphed_data = {
                "time": time.time(),
                "block": self.metagraph.block.item(),
                "uid_data": {
                    str(uid): uid_data[str(uid)]["loss_avg"] for uid in uids
                },
                "weight_data": {str(uid): self.weights[uid].item() for uid in uids},
                "load_model_perf_log": load_model_perf_str,
                "compute_model_perf_log": compute_loss_perf_str,
            }
            bt.logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json},
                step=self.global_step,
            )

    async def run(self):
        """Runs the validator loop, which continuously evaluates models and sets weights."""
        while True:
            try:
                while (
                        (self.metagraph.block.item() - self.last_epoch)
                        < self.config.blocks_per_epoch
                ):
                    self.current_block = self.metagraph.block.item()
                    await self.try_run_step(ttl=60 * 20)
                    await self.try_sync_metagraph(ttl=60)
                    self.save_state()
                    bt.logging.debug(
                        f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch."
                    )
                    self.global_step += 1

                if not self.config.dont_set_weights and not self.config.offline:
                    await self.try_set_weights(ttl=60)
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info(
                    "KeyboardInterrupt caught, gracefully closing the wandb run..."
                )
                if self.wandb_run:
                    self.wandb_run.finish()
                exit()

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )


if __name__ == "__main__":
    asyncio.run(Validator().run())
