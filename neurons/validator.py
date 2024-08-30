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

# Due to the implementation of disable_progress_bars(), this has to be the first import+call in the application relating to huggingface
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

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
from packaging.version import Version

import wandb
import constants
import dataset
import validation
from model import model_utils, competitions
from model.data import ModelId, ModelMetadata
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
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
from utilities import utils, btlite
from utilities.perf_monitor import PerfMonitor
from utilities.mathutils import *

TRANSFORMERS_VERSION_MIN     = "4.41.2"
TRANSFORMERS_VERSION_OPTIMAL = "4.44.0"

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Container:
    '''Empty container object'''
    pass

class ModelIssue(Exception):
    '''
    Exception class to signal issues with models preventing evaluation.
    '''
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

        self.force_eval_until_uid_last = None
        self.last_weights_set = time.time()
        with self.state_lock:
            if 'version' in state and state['version'] == constants.__spec_version__:
                self.hall_of_fame = state.pop("hall_of_fame", {})
                self.competitions = state.pop("competitions", {})
                tracker_state = state.pop("tracker", {})
                self.cstate = state.pop('cstate', {})
                bt.logging.info("State loaded successfully")
                self.force_eval_until_uid_last = state.pop("force_eval_until_uid_last", None)
                self.last_weights_set = state.pop("last_weights_set", time.time())
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
                'force_eval_until_uid_last': self.force_eval_until_uid_last,
                'last_weights_set': self.last_weights_set,
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
        self.subtensor = btlite.get_subtensor(config=self.config)
        self.subtensor_lock = threading.RLock()
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid, lite=False)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        self.uid = None
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
        self.last_weights_set = time.time() # Don't set weights early

        # model_tracker tracks which miner is using which model id.
        self.state_lock = threading.RLock()
        self.model_tracker = ModelTracker()

        # Load or initialize internal state
        self.load_state()

        # Setup a miner iterator to ensure we update all miners.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        bt.logging.trace("Starting ModelUpdater")
        self.model_updater = ModelUpdater(
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
            comps=self.competitions
        )

        # Create a metagraph lock to avoid cross thread access issues in the update and clean loop.
        self.metagraph_lock = threading.RLock()

        # Initialize the update thread
        self.stop_event = threading.Event()
        bt.logging.trace("Starting update thread")
        self.update_thread_ts = time.time()
        self.update_thread = threading.Thread(target=self.update_thread_func, daemon=True)
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
                if meta is not None and cname == meta.id.competition:
                    bt.logging.info(f"Adding {uid} to competition {cname}")
                    if uid not in cinfo['uids_pool'] and uid not in cinfo['uids_pending']:
                        cinfo['uids_pending'].append(uid)
                else:
                    if uid in cinfo['uids_pool']:
                        cinfo['uids_pool'].remove(uid)
                    if uid in cinfo['uids_pending']:
                        cinfo['uids_pending'].remove(uid)

    def get_metagraph(self, n_retries=3):
        for i in range(n_retries):
            try:
                metagraph = self.subtensor.metagraph(self.config.netuid, lite=False)
                if metagraph is not None:
                    return metagraph
            except Exception as e:
                bt.logging.warning(f"Failed to get metagraph {i+1}/{n_retries}: {e}\n{traceback.format_exc()}")
            bt.logging.info("Reconnecting subtensor")
            st = btlite.get_subtensor(config=self.config)
            with self.subtensor_lock:
                self.subtensor = st

        bt.logging.error(f"Failed to get metagraph {n_retries} times, giving up")
        return None

    def update_chain(self):
        now = time.time()
        if self.last_chain_update is not None and \
                (now - self.last_chain_update) < constants.CHAIN_UPDATE_INTERVAL:
            return

        # We would like to run chain interaction commands with a timeout, but there is no
        # straightforward method to do that using threads (as all options with a timeout
        # leave the thread running). We do not want to use subprocesses either.
        # Therefore, we run blocking in this thread. It would be nice if the bittensor library
        # timeout properties would be configurable (there are some hard-coded retries etc).
        # The main thread can monitor whether the thread performing these updates is still
        # alive, and bail out if that is not the case.
        new_metagraph = self.get_metagraph()
        if new_metagraph is None:
            return False

        with self.metagraph_lock:
            self.metagraph = copy.deepcopy(new_metagraph)
            self.model_tracker.on_hotkeys_updated(set(self.metagraph.hotkeys))

        self.last_chain_update = now

        bt.logging.warning(f"Synced metagraph with {len(self.metagraph.neurons)} neurons")

        # Determine top miners according to other valis
        top_miner_uids = set(utils.list_top_miners(new_metagraph))
        bt.logging.info(f"Top miners: {top_miner_uids}")
        top_uids = top_miner_uids - self.get_all_active_uids()

        # Determine for which top uids to force retry
        now = time.time()
        top_uids_to_eval = []
        for uid in top_uids:
            if now - self.uid_last_retried_ts.get(uid,0) > constants.TOP_MODEL_RETRY_INTERVAL:
                top_uids_to_eval.append(uid)
                self.uid_last_retried_ts[uid] = now

        # We need all models te be re-evaluated periodically, to make sure they
        # can win on changing competition parameters or advantage factors.
        # This is synchronized between validators by using UTC time, in order
        # to keep their weights in sync as much as possible. By testing only a
        # few extra models at a time, the weight setting interval is kept low
        # (previously there was a "test all models" event which took way longer
        # than a regular eval loop).
        now = int(time.time())
        interval = constants.GEN_MODEL_RETRY_INTERVAL
        force_eval_frac = float(now%interval)/interval
        force_eval_until_uid = int(force_eval_frac*constants.SUBNET_N_UIDS)
        if self.force_eval_until_uid_last is None:
            # note regarding negative modulo in python: -1%10==9
            self.force_eval_until_uid_last = (force_eval_until_uid-constants.MODEL_RETRY_MAX_N_PER_LOOP)%constants.SUBNET_N_UIDS
        n_forced_periodic = 0
        n_unforced_updated = 0

        bt.logging.info(f"Forcing re-evaluation of models with {self.force_eval_until_uid_last} < UID <= {force_eval_until_uid}")

        # Retrieve chain metadata, download new/forced models
        start_uid = random.randint(0, max(0,len(new_metagraph.hotkeys)-1))   # Pick random UID to start from
        cur_uid = start_uid
        while True:
            self.update_thread_ts = time.time()
            hotkey = new_metagraph.hotkeys[cur_uid]

            # Sync the model, if necessary.
            try:
                # check if the uid is in the range of models to force
                force_periodic = False
                if n_forced_periodic >= constants.MODEL_RETRY_MAX_N_PER_LOOP:
                    # Already added enough models to force-retry.
                    # The risk of systematically skipping certain UIDs is small, as
                    # the range to include depends on current time, and validator
                    # loops run freely.
                    pass
                elif self.force_eval_until_uid_last <= force_eval_until_uid:
                    if self.force_eval_until_uid_last < cur_uid <= force_eval_until_uid:
                        force_periodic = True
                else:
                    if cur_uid > self.force_eval_until_uid_last or cur_uid <= force_eval_until_uid:
                        force_periodic = True

                force = cur_uid in top_uids_to_eval or force_periodic

                if not force and n_unforced_updated >= constants.MODEL_UNFORCED_N_PER_LOOP:
                    bt.logging.debug(f"Skipped UID {cur_uid}/{hotkey}, already {n_unforced_updated} unforced model updates")
                else:
                    with self.subtensor_lock:
                        metadata = bt.extrinsics.serving.get_metadata(self.subtensor, self.config.netuid, hotkey)
                        if metadata is not None:
                            metadata = ModelMetadata.parse_chain_data(metadata)

                    updated = asyncio.run(
                        self.model_updater.sync_model(hotkey, metadata, force=force)
                    )
                    remark = ''
                    if updated:
                        self.add_uid_to_competition(cur_uid, hotkey)
                        if not force:
                            n_unforced_updated += 1
                        if force_periodic:
                            n_forced_periodic += 1
                            remark = ' (included for periodic evaluation)'
                    bt.logging.debug(f"Visited UID {cur_uid}/{hotkey}, updated={updated}, commitment: {metadata.id.format_label() if metadata else '---'}{remark}")
            except Exception as e:
                bt.logging.error(
                    f"Failed to sync model for UID {cur_uid}: {type(e).__name__} {e} \n {traceback.format_exc()}"
                )

            cur_uid = (cur_uid + 1) % len(new_metagraph.hotkeys)
            if cur_uid == start_uid:
                break

        self.force_eval_until_uid_last = force_eval_until_uid

    def update_dynamic_config(self):
        now = time.time()
        if self.last_cfg_fetch is not None and \
                (now - self.last_cfg_fetch) < constants.CFG_FETCH_INTERVAL:
            return
        self.last_cfg_fetch = now

        # Competition info
        url = constants.COMPETITIONS_URL
        comps = None
        if self.uid is not None:
            # Check if a dedicated competitions-{uid}.json is available.
            # This is used to allow targeted tweaks e.g. in case of issues with transformer overrides.
            vali_url = url.replace('.json',f'-{self.uid}.json')
            comps = competitions.load_competitions(vali_url,warn_failure=False)
        if comps is None:
            # Load regular competition info.
            comps = competitions.load_competitions(url)
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

    def update_thread_func(self):
        """
        Updates all remote metadata: chain, competitions, hall of famae.
        """

        # Timestamp when we last retried models with incentive we've already dropped
        self.uid_last_retried_ts = dict()
        # Timestamp when we last fetched hall of fame
        self.last_cfg_fetch = None
        # Timestamp when we last updated chain
        self.last_chain_update = None

        while not self.stop_event.is_set():
            try:
                self.update_dynamic_config()
            except Exception as e:
                bt.logging.error(f"Failed to update dynamic config: {e} \n {traceback.format_exc()}")

            try:
                self.update_chain()
            except Exception as e:
                bt.logging.error(f"Failed to update chain data: {e} \n {traceback.format_exc()}")

            # Regular sleep, we are running in a separate thread
            self.update_thread_ts = time.time()
            time.sleep(60)

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
        if self.last_weights_set is not None and time.time() - self.last_weights_set < constants.WEIGHT_SET_MIN_INTERVAL:
            return

        # Prepare and log weights
        self.weights.nan_to_num(0.0)
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

        # always get a new subtensor instance for weight setting
        st = btlite.get_subtensor()
        if st is None:
            bt.logging.error(f'Could not create subtensor, cannot set weights')
            return

        try:
            bt.logging.warning(f"Setting weights.")
            call = btlite.set_weights_retry(
                subtensor=st,
                hotkey=self.wallet.hotkey,
                uids=self.metagraph.uids,
                netuid=constants.SUBNET_UID,
                weights=self.weights[:len(self.metagraph.uids)],
                wait_for_inclusion=True,
                version_key=constants.weights_version_key,
            )
            ret,msg = await asyncio.wait_for(call, ttl)
            if ret:
                self.last_weights_set = time.time()
                bt.logging.warning(f"Finished setting weights at {self.last_weights_set}.")
            else:
                bt.logging.warning(f"Failed to set weights: {msg}")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")
        except Exception as e:
            bt.logging.warning(f"Failed to set weights: {e}, {traceback.format_exc()}")


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
        batches_max_token_id = None
        if 'tokenizer' in cinfo:
            # Competition-wide (forced or default) tokenizer --> fixed sequence length
            batches = dataloader.tokenize(
                    cinfo['tokenizer'],
                    max_len=cinfo.get('max_sequence_len', constants.MAX_SEQUENCE_LEN),
                    max_invalid=cinfo.get('max_tokenize_fails', constants.MAX_TOKENIZE_FAILS)
            )
            batches_max_token_id = max(
                [max(b[0]) for b in batches if b is not None]
            )


        # Compute model losses on batches.
        losses_per_uid = {uid: None for uid in uids_pool}
        losses_pt_per_uid = {uid: None for uid in uids_pool}
        avg_sample_len_per_uid = {uid: None for uid in uids_pool}
        uid_to_label = {uid: '' for uid in uids_pool}
        uid_to_block = {uid: 1<<31 for uid in uids_pool}
        n_evaluated = 0
        for uid in uids_pool:
            bt.logging.trace(f"Computing model losses for uid {uid}.")
            metadata = self.get_uid_metadata(uid)

            losses_per_uid[uid] = [math.inf]*n_batches
            losses_pt_per_uid[uid] = losses_per_uid[uid].copy()
            if metadata is None:
                bt.logging.debug(f"Unable to load metadata for {uid}. Setting loss to infinity.")
                continue
            try:
                uid_to_block[uid] = metadata.block if metadata.block is not None else 1<<31
                uid_to_label[uid] = metadata.id.format_label()
                bt.logging.debug(f"Evaluating uid {uid} ({uid_to_label[uid]}) from block {uid_to_block[uid]}")
                # Get model tokenizer if no competition-wide tokenizer is set
                mdl_batches = batches
                max_token_id = batches_max_token_id
                model_path = disk_utils.get_local_model_snapshot_dir(
                        self.local_store.base_dir,
                        metadata.hotkey,
                        metadata.id) if metadata.path is None else metadata.path
                tokenizer_json = os.path.join(model_path,'tokenizer.json')
                if not os.path.exists(tokenizer_json):
                    # Assume tokenizer.json indicates an embedded tokenizer is available.
                    if mdl_batches is None:
                        raise ModelIssue(f'No default tokenizer and no model tokenizer available')
                elif mdl_batches is None or cinfo.get('free_tokenizer',False):
                    max_len = cinfo.get('max_sequence_len', None)
                    if max_len is None:
                        raise ModelIssue(f"Unable to determine max sequence length")
                    try:
                        new_mdl_batches = None
                        new_mdl_batches = dataloader.tokenize(
                                model_path,
                                max_len=max_len,
                                max_invalid=cinfo.get('max_tokenize_fails', constants.MAX_TOKENIZE_FAILS)
                        )
                        mdl_batches = new_mdl_batches
                        max_token_id = max(
                            [max(b[0]) for b in mdl_batches if b is not None]
                        )
                        bt.logging.info("Using model-supplied tokenizer")
                    except Exception as e:
                        if new_mdl_batches is None:
                            if mdl_batches is None:
                                raise ModelIssue(f'No default tokenizer and no model tokenizer available: {e}')
                            bt.logging.info(f"Using default tokenizer {cinfo.get('tokenizer', 'unknown')}, because {e}")

                if mdl_batches is None:
                    # We should never arrive here
                    raise Exception("No tokenizer available (no default and not supplied in model)")


                losses,losses_pt,avg_sample_len = utils.run_in_subprocess(
                    functools.partial(
                        check_and_compute_losses,
                        local_store=self.local_store,
                        metadata=metadata,
                        competition_info=cinfo,
                        batches=mdl_batches,
                        max_token_id=max_token_id,
                        device=self.config.device,
                    ),
                    ttl=360,
                    mode="spawn",
                    expected_errors={"ModelIssue"},
                )

                n_evaluated += 1

                losses_per_uid[uid] = losses
                losses_pt_per_uid[uid] = losses_pt
                avg_sample_len_per_uid[uid] = avg_sample_len
                bt.logging.debug(f"Losses for uid:{uid}, per token: {naninf_mean(losses_pt):.03f} +- {naninf_std(losses_pt):.03f}, sum {naninf_mean(losses):.01f} +- {naninf_std(losses):.01f}, avg sample len: {avg_sample_len:.01f}")

            except ModelIssue as e:
                bt.logging.info(
                    f'Model issue for uid {uid}, disqualifying: {e}'
                )
            except Exception as e:
                bt.logging.error(
                    f"Error in eval loop: {e}. Setting losses for uid: {uid} to infinity.\n{traceback.format_exc()}"
                )
                if transformers.__version__ != TRANSFORMERS_VERSION_OPTIMAL:
                    bt.logging.error(f'Please run with transformers version {TRANSFORMERS_VERSION_OPTIMAL} (currently running {transformers.__version__}) before reporting issues.')

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
        skew_factor = cinfo.get('weight_skew_factor',constants.WEIGHT_SKEW_FACTOR)
        model_weights = win_rate**skew_factor
        weight_sum = np.sum(model_weights)
        if weight_sum:
            model_weights /= weight_sum
        model_weights = {uid: weight for uid, weight in zip(uids_pool, model_weights)}

        # Sort models by weight / win_rate, keep pool_size entries
        pool_size = cinfo.get('pool_size', constants.DEFAULT_POOL_SIZE)
        win_rate_indices = win_rate.argsort()[-pool_size:]
        new_uids_pool = [uids_pool[i] for i in win_rate_indices]
        bt.logging.warning(f'selected {pool_size} winning models: {new_uids_pool}')

        # Update state: weights and which uids to keep for next run
        with self.state_lock:
            self.cstate[cname]['uids_weight'] = model_weights
            self.cstate[cname]['uids_pool'] = new_uids_pool

        win_matrix = win_info.get('matrix', None)
        if win_matrix is not None:
            bt.logging.info(f"Competition {cname} result:")
            self.print_win_matrix(win_info['matrix'], uid_to_label, competition=cname)

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
                "n_samples": naninf_count(losses_per_uid[uid]),
                "n_inf": np.sum(np.isinf(losses_per_uid[uid])),
                "avg_sample_len": avg_sample_len_per_uid[uid],
                "loss_pt_avg": naninf_mean(losses_pt_per_uid[uid]),
                "loss_pt_std": naninf_std(losses_pt_per_uid[uid]),
                "loss_sum_avg": naninf_mean(losses_per_uid[uid]),
                "loss_sum_std": naninf_std(losses_per_uid[uid]),
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

    def print_win_matrix(self, matrix, uid_to_label={}, show_delta_loss=False, competition='?'):
        if show_delta_loss:
            title = f"Win matrix compt {competition}, true wins/adv wins/avg delta loss"
        else:
            title = f"Win matrix compt {competition}, true wins/adv wins"
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
        table.add_column("avg_slen", style="magenta")
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
                    f"{d['avg_sample_len']:.01f}",
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
                self.current_block = self.metagraph.block.item()
                await self.try_run_step(ttl=60 * 60)
                self.save_state()
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

def check_and_compute_losses(
        local_store=None,
        metadata=None,
        competition_info=None,
        batches=None,
        max_token_id=None,
        device=None,
    ):
    cinfo = competition_info
    model_i = local_store.retrieve_model(metadata.hotkey, metadata.id, path=metadata.path)
    mdl_allowed, reason = competitions.validate_model_constraints(model_i.pt_model, cinfo)
    if not mdl_allowed:
        raise ModelIssue(f"Model violates competition {cname} constraints: {reason}")
    allow_sliced = False
    model_type = type(model_i.pt_model).__name__
    if 'Sliced' in model_type:
        # Test the exact model type name to check whether slicing is allowed by config:
        allow_sliced = model_type in cinfo['model_types']

    embed_size = None
    try:
        embed_size = model_i.pt_model.model.embed_tokens.weight.shape[0]
    except Exception as e:
        # Currently supported models should have the queried parameter, but in case they don't, just skip this check.
        bt.logging.warning(f'could not find embed size, skipping check: {e}')

    if embed_size and max_token_id>=embed_size:
        raise ModelIssue(f"Vocabulary size mismatch between tokenizer and model: {max_token_id} >= {embed_size}")

    losses = validation.compute_losses(model_i.pt_model,allow_sliced,batches,device)
    losses_pt = [loss_sum / len(batch[0]) if batch is not None else math.inf for loss_sum, batch in zip(losses, batches)]
    sample_lengths = [len(batch[0]) for batch in batches if batch is not None]
    avg_sample_length = 0 if len(sample_lengths) == 0 else np.mean(sample_lengths)

    return losses,losses_pt,avg_sample_length


if __name__ == "__main__":
    if Version(transformers.__version__) < Version(TRANSFORMERS_VERSION_MIN):
        bt.logging.error(f"Transformers version >= {TRANSFORMERS_VERSION_MIN} required")
        sys.exit()

    asyncio.run(Validator().run())
