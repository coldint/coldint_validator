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
import orjson
import math
import pickle
import time
import torch
import random
import asyncio
import numpy as np
import requests
import sys
import transformers

import wandb
import constants
import dataset
import validation
from model import model_utils, competitions
from model.data import ModelId
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.disk import utils as disk_utils
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

TRANSFORMERS_VERSION_MIN     = "4.41.2"

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Container:
    '''Empty container object'''
    pass

class Validator:
    STATE_FILENAME = "validator_state.json"
    BENCHMARK_FILENAME = "benchmark.json"

    def state_path(self) -> str:
        return os.path.join(self.config.model_dir, "vali-state")

    def load_state(self):
        # Updated by model updater thread, used by evaluation thread
        self.hall_of_fame = {}
        self.competitions = {}

        # Competition state, only used by evaluation thread
        self.cstate = {}

        state = {}
        state_fn = os.path.join(self.state_path(), Validator.STATE_FILENAME)
        if os.path.exists(state_fn):
            try:
                with open(state_fn, 'rb') as f:
                    state = orjson.loads(f.read())
            except Exception as e:
                bt.logging.info(f"Invalid state file: {e}")

        with self.state_lock:
            if 'version' in state and state['version'] == constants.__spec_version__:
                self.hall_of_fame = state.pop("hall_of_fame", {})
                self.competitions = state.pop("competitions", {})
                tracker_state = state.pop("tracker", {})
                self.cstate = state.pop('cstate', {})
                bt.logging.info("State loaded successfully")
            else:
                bt.logging.info("State version incompatible, starting with clean state")
                tracker_state = {}

        self.model_tracker.set_state(tracker_state)

    def save_state(self):
        state_dir = self.state_path()
        os.makedirs(state_dir, exist_ok=True)
        with self.state_lock:
            state = {
                'version': constants.__spec_version__,
                'competitions': self.competitions,
                'hall_of_fame': self.hall_of_fame,
                'tracker': self.model_tracker.get_state(),
                'cstate': self.cstate,
            }
        try:
            with open(os.path.join(self.state_path(), Validator.STATE_FILENAME), 'wb') as f:
                f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2|orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY))
        except Exception as e:
            bt.logging.warning(f"Failed to save state: {e}")

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

        # Dont log to wandb if offline.
        self.wandb_run = None
        if self.config.wandb.on and not self.config.offline:
            self.new_wandb_run()

        # Weights and step info
        self.weights = torch.zeros(constants.SUBNET_N_UIDS)
        self.run_step_count = 0
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        # model_tracker tracks which miner is using which model id.
        self.state_lock = threading.RLock()
        self.model_tracker = ModelTracker()

        # Load or initialize internal state
        self.load_state()

        # Setup a miner iterator to ensure we update all miners.
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
            comps=self.competitions
        )

        # Create a metagraph lock to avoid cross thread access issues in the update and clean loop.
        self.metagraph_lock = threading.RLock()

        # Initialize the update thread
        self.stop_event = threading.Event()
        bt.logging.trace("Starting update_models thread")
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

        # Initialize the cleaner thread to remove outdated models
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

    def get_all_active_uids(self):
        ret = []
        with self.state_lock:
            for cname, cinfo in self.cstate.items():
                ret.extend(cinfo['uids_pool'])
                ret.extend(cinfo['uids_pending'])
        return set(ret)

    def add_uid_to_competition(self, uid, hotkey):
        """
        Add uid to the competition pool which it participates in, delete it from others.
        """
        meta = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        if meta is None:
            bt.logging.warning(f"Metadata for {uid}/{hotkey} not available")
        with self.state_lock:
            for cname, cinfo in self.cstate.items():
                if cname == meta.id.competition:
                    bt.logging.info(f"Adding {uid} to competition {cname}")
                    if uid not in cinfo['uids_pool'] and uid not in cinfo['uids_pending']:
                        cinfo['uids_pending'].append(uid)
                else:
                    if uid in cinfo['uids_pool']:
                        cinfo['uids_pool'].remove(uid)
                    if uid in cinfo['uids_pending']:
                        cinfo['uids_pending'].remove(uid)

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
        uids_to_add = top_miner_uids - self.get_all_active_uids()

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

                self.add_uid_to_competition(uid, hotkey)
                bt.logging.debug(
                    f"Retrying evaluation for previously discarded model with incentive for UID={uid}."
                )
            except Exception:
                bt.logging.debug(
                    f"Failure in update loop for UID={uid} during top model check. {traceback.format_exc()}"
                )


    def update_dynamic_config(self):
        now = time.time()
        if self.last_cfg_fetch is not None and \
                (now - self.last_cfg_fetch) < constants.CFG_FETCH_INTERVAL:
            return
        self.last_cfg_fetch = now

        # Competition info
        comps = competitions.load_competitions(constants.COMPETITIONS_URL)
        if comps is not None:
            with self.state_lock:
                self.competitions = comps
                self.model_updater.set_competitions(self.competitions)

        # Hall of fame
        try:
            req = requests.get(constants.HOF_URL)
            req.raise_for_status()
            with self.state_lock:
                self.hall_of_fame = req.json()
            bt.logging.info(f"Fetched hall of fame content, containing {len(self.hall_of_fame)} entries")
        except Exception as e:
            bt.logging.error(f"Failed to fetch hall of fame: {e}")

        self.save_state()

    def update_models(self):
        """
        Updates the models in the local store based on the latest metadata from the chain.
        Periodically fetch hall of fame / competition config json.
        """

        # Track how recently we updated each uid from sequential iteration.
        uid_last_checked_sequential = dict()
        # Track how recently we checked the list of top models.
        self.last_checked_top_models_time = None
        # Track how recently we retried a model with incentive we've already dropped.
        self.uid_last_retried_evaluation = dict()
        # Track when we last fetched hall of fame
        self.last_cfg_fetch = None

        while not self.stop_event.is_set():
            try:
                self.update_dynamic_config()
                self.check_top_models()

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
                    self.add_uid_to_competition(next_uid, hotkey)

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
                uids_to_keep = self.get_all_active_uids()

                with self.metagraph_lock:
                    # Injected models might not be in metagraph
                    hotkeys_to_keep = set([
                        self.metagraph.hotkeys[uid]
                            for uid in uids_to_keep if uid < len(self.metagraph.hotkeys)
                    ])

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
                bt.logging.error(f"Error in clean loop: {e}, {traceback.format_exc()}")

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

    def update_weights(self):
        '''
        Update self.weights, based on internal cstate
        Up to constants.REWARDS_MAX_FRACTION part of the total weight will be based
        on the hall of fame rewards settings.
        '''
        new_weights = torch.zeros_like(self.weights)
        reward_weights, reward_sum = self.get_reward_weights()

        # Scale model weights down by (1 - reward_sum)
        for cname, cparams in self.competitions.items():
            if cname not in self.cstate:
                bt.logging.warning(f"No evaluations in competition {cname}")
                continue

            for uid, weight in self.cstate[cname]['uids_weight'].items():
                new_weights[uid] = (1 - reward_sum) * cparams['reward'] * weight

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
            with open(self.BENCHMARK_FILENAME, 'rb') as f:
                d = {int(k): v for k, v in orjson.loads(f.read()).items()}
            bt.logging.info(f"Loaded benchmark config with {len(d)} items")
            return d
        except Exception as e:
            bt.logging.warning(f"Failed to load benchmark config: {e}")
        return {}

    async def run_step(self):
        bt.logging.debug(f'run_step() @ current block {self.current_block}')
        self.inject_models()

        # Collect uid evaluation data here
        self.step_uid_log = dict()

        # Currently, all competitions use the same dataset.
        # Fetch samples that are shared between all competitions
        dataloader = dataset.SubsetFineWebEdu2Loader(
            batch_size=1,
            num_pages=0,
            tokenizer=None,
            pack=False
        )
        samples = dataloader.fetch_data_to_rows(constants.n_eval_pages)
        if len(samples) == 0:
            bt.logging.warning(f"No samples to eval. Waiting one minute before retrying.")
            await asyncio.sleep(60)
            return

        n_models_evaluated = 0
        for cname in self.competitions:
            n_models_evaluated += self.run_step_for_competition(cname, dataloader)

        if n_models_evaluated == 0:
            bt.logging.debug("No uids to eval. Waiting 2 minutes to download some models.")
            await asyncio.sleep(120)
            return

        self.update_weights()

        self.save_state()

        # Log to screen and wandb.
        pages_desc = [f'{cfg_name}_{num_rows}_{split}' for cfg_name, num_rows, split in dataloader.pages]
        self.log_step(pages_desc)

        # Increment the number of completed run steps by 1
        self.run_step_count += 1

    def run_step_for_competition(self, cname, dataloader):
        """
        Run step for one of the competitions
        Return number of uids evaluated
        """
        with self.state_lock:
            # Initialize competition state
            if cname not in self.cstate:
                self.cstate[cname] = dict(
                    uids_pool=[],
                    uids_pending=[],
                )
            cstate = self.cstate[cname]
            cstate['uids_pool'] = list(set(cstate['uids_pool']) | set(cstate['uids_pending']))
            cstate['uids_pending'] = []
            uids_pool = cstate['uids_pool']

        if len(uids_pool) == 0:
            with self.state_lock:
                cstate['uids_weight'] = {}
            bt.logging.debug(f"No miners participating in competition {cname}")
            return 0

        bt.logging.debug(f"Evaluating competition {cname} with uids {uids_pool} on {len(dataloader.buffer)} samples")

        cinfo = self.competitions[cname]

        # Competition-wide tokenizer
        n_batches = len(dataloader.buffer)
        batches = None
        if 'tokenizer' in cinfo:
            batches = dataloader.tokenize(cinfo['tokenizer'], max_len=constants.MAX_SEQUENCE_LEN)

        # Compute model losses on batches.
        losses_per_uid = {uid: None for uid in uids_pool}
        losses_pt_per_uid = {uid: None for uid in uids_pool}
        uid_to_label = {uid: '' for uid in uids_pool}
        uid_to_block = {uid: 1<<31 for uid in uids_pool}
        n_evaluated = 0
        for uid in uids_pool:
            bt.logging.trace(f"Computing model losses for uid {uid}.")
            metadata = self.get_uid_metadata(uid)

            losses = [math.inf]*n_batches
            losses_pt = losses.copy()
            if metadata is not None:
                try:
                    uid_to_block[uid] = metadata.block if metadata.block is not None else 1<<31
                    uid_to_label[uid] = metadata.id.format_label()
                    bt.logging.debug(f"Evaluating uid {uid} ({uid_to_label[uid]}) from block {uid_to_block[uid]}")

                    # Get model tokenizer if no competition-wide tokenizer is set
                    mdl_batches = batches
                    if mdl_batches is None:
                        model_path = disk_utils.get_local_model_snapshot_dir(
                                self.local_store.base_dir,
                                metadata.hotkey,
                                metadata.id) if metadata.path is None else metadata.path
                        mdl_batches = dataloader.tokenize(model_path, max_len=constants.MAX_SEQUENCE_LEN)

                    model_i = self.local_store.retrieve_model(metadata.hotkey, metadata.id, path=metadata.path)
                    mdl_allowed, reason = competitions.validate_model_constraints(model_i.pt_model, cinfo)
                    if mdl_allowed:
                        losses = utils.run_in_subprocess(
                            functools.partial(
                                validation.compute_losses,
                                model_i.pt_model,
                                mdl_batches,
                                self.config.device
                            ),
                            ttl=360,
                            mode="spawn",
                        )
                        losses_pt = [loss_sum / len(batch[0]) for loss_sum, batch in zip(losses, mdl_batches)]
                        n_evaluated += 1
                    else:
                        bt.logging.info(f"Model for uid {uid} violates competition {cname} constraints: {reason}")

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
            losses_pt_per_uid[uid] = losses_pt
            bt.logging.debug(f"Losses for uid:{uid}, per token: {np.nanmean(losses_pt):.03f} +- {np.nanstd(losses_pt):.03f}, sum {np.nanmean(losses):.01f} +- {np.nanstd(losses):.01f}")

        win_info = validation.compute_wins(
                losses_per_uid,
                uid_to_block,
                self.current_block,
                cinfo.get('advantage_initial', constants.advantage_initial),
                cinfo.get('advantage_decay', constants.advantage_decay_per_epoch),
        )
        if 'win_rate' not in win_info:
            bt.logging.warning("compute_wins() returned no result")
            return 0

        # Skew weight distribution towards models with high win_rate
        win_rate = np.array([
            win_info['win_rate'][uid] if uid in win_info['win_rate'] else 0
                for uid in uids_pool
        ])
        model_weights = win_rate**constants.WEIGHT_SKEW_FACTOR
        model_weights /= np.sum(model_weights)
        model_weights = {uid: weight for uid, weight in zip(uids_pool, model_weights)}

        # Sort models by weight / win_rate, keep pool_size entries
        pool_size = cinfo.get('pool_size', constants.DEFAULT_POOL_SIZE)
        new_uids_pool = list(sorted(model_weights, key=model_weights.get, reverse=True)[:pool_size])

        # Update state: weights and which uids to keep for next run
        with self.state_lock:
            self.cstate[cname]['uids_weight'] = model_weights
            self.cstate[cname]['uids_pool'] = new_uids_pool

        win_matrix = win_info.get('matrix', None)
        if win_matrix is not None:
            bt.logging.info(f"Competition {cname} result:")
            self.print_win_matrix(win_info['matrix'], uid_to_label)

        # Update step log
        wins = win_info.get('wins', {})
        win_rate = win_info.get('win_rate', {})
        advantage_factors = win_info.get('advantage_factors', {})
        for uid in uids_pool:
            self.step_uid_log[uid] = {
                "uid": uid,
                "competition": cname,
                "label": uid_to_label.get(uid, ''),
                "block": uid_to_block.get(uid, 1<<31),
                "losses": losses_per_uid[uid],
                "loss_pt_avg": np.nanmean(losses_pt_per_uid[uid]),
                "loss_pt_std": np.nanstd(losses_pt_per_uid[uid]),
                "loss_sum_avg": np.nanmean(losses_per_uid[uid]),
                "loss_sum_std": np.nanstd(losses_per_uid[uid]),
                "adv_factor": 100*(1-advantage_factors.get(uid,1)),
                "win_rate": win_rate.get(uid, 0),
                "win_total": wins.get(uid, 0),
                "win_matrix_row": win_matrix.get(uid, None) if win_matrix else None
            }

        return n_evaluated

    def get_uid_metadata(self, uid):
        metadata = Container()
        if uid in self.benchmark_cfg:
            # Model data from dynamic config
            bcfg = self.benchmark_cfg[uid]
            metadata.hotkey = bcfg.get("hotkey", "xxx")
            metadata.block = bcfg.get("block", 1<<31)
            metadata.path = bcfg.get('path', 'please specify path')
            metadata.id = ModelId.dummy(bcfg.get('label', os.path.split(metadata.path)[-1]))
        elif uid < len(self.metagraph.hotkeys):
            # Model from chain
            metadata.hotkey = self.metagraph.hotkeys[uid]
            chain_data = self.model_tracker.get_model_metadata_for_miner_hotkey(metadata.hotkey)
            metadata.id = chain_data.id
            metadata.block = chain_data.block
            metadata.path = None
        else:
            metadata = None
        return metadata

    def inject_models(self):
        self.benchmark_cfg = self.load_benchmark_config()
        with self.state_lock:
            for uid, binfo in self.benchmark_cfg.items():
                competition = binfo.get('competition', '')
                if competition not in self.cstate:
                    bt.logging.info(f"Injected model {uid} competition '{competition}' unknown")
                    continue
                ci = self.cstate[competition]
                if uid not in ci['uids_pool'] and uid not in ci['uids_pending']:
                    ci['uids_pending'].append(uid)

    def print_win_matrix(self, matrix, uid_to_label={}, show_delta_loss=False):
        if show_delta_loss:
            title = "Model win matrix, true wins/adv wins/avg delta loss"
        else:
            title = "Model win matrix, true wins/adv wins"
        table = Table(title=title)
        table.add_column("win \ lose", justify="right", style="cyan", no_wrap=True)
        for uid in matrix:
            table.add_column(f'UID {uid}')
        for uid_a,row in matrix.items():
            label = uid_to_label.get(uid_a, '')
            vals = [f'{label} UID {uid_a}']
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

    def log_step(self, pages):
        """Logs the results of the step to the console and wandb (if enabled)."""
        # Build step log
        step_log = {
            "timestamp": time.time(),
            'block': self.current_block,
            "pages": pages,
            "uids": [int(uid) for uid in self.step_uid_log.keys()],
            "competitions": self.competitions,
            "uid_data": {},
        }
        for uid, info in self.step_uid_log.items():
            info['weight'] = self.weights[uid].item()
            step_log['uid_data'][str(uid)] = info

        if self.config.save_step_json:
            try:
                with open(self.config.save_step_json, 'wb') as f:
                    f.write(orjson.dumps(step_log, option=orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY))
            except Exception as e:
                bt.logging.warning(f"Failed to write step json to {self.config.save_step_json}: {e}")

        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("compt", style="magenta")
        table.add_column("avg_pt_loss", style="magenta")
        table.add_column("avg_sum_loss", style="magenta")
        table.add_column("adv_factor(%)", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid, d in step_log['uid_data'].items():
            try:
                table.add_row(
                    f"{uid} {d.get('label','')}",
                    str(d['competition']),
                    f"{d['loss_pt_avg']:.04f}+-{d['loss_pt_std']:.04f}",
                    f"{d['loss_sum_avg']:.01f}+-{d['loss_sum_std']:.01f}",
                    f"{d['adv_factor']:.03f} of {100*constants.advantage_initial:.03f}",
                    str(round(d["win_rate"], 4)),
                    str(d["win_total"]),
                    str(round(d['weight'], 4)),
                    str(d["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

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

            original_format_json = orjson.dumps(step_log, option=orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY).decode()
            uids = step_log["uids"]
            uid_data = step_log["uid_data"]

            # Create a new dictionary with the required format
            graphed_data = {
                "time": time.time(),
                "block": self.metagraph.block.item(),
                "uid_data": {
                    str(uid): uid_data[str(uid)]["loss_sum_avg"] for uid in uids
                },
                "weight_data": {str(uid): self.weights[uid].item() for uid in uids},
            }
            bt.logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json},
                step=self.global_step,
            )

    async def run(self):
        """Runs the validator loop, which continuously evaluates models and sets weights."""
        # Give check_top_models some time
        await asyncio.sleep(60)
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
    if transformers.__version__ < TRANSFORMERS_VERSION_MIN:
        bt.logging.error(f"Transformers version >= {TRANSFORMERS_VERSION_MIN} required")
        sys.exit()

    asyncio.run(Validator().run())
