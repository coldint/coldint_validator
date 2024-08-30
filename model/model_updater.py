import bittensor as bt
from typing import Optional
import constants
import os
from model import competitions
from model.model_utils import get_hash_of_two_strings
from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.storage.disk import utils
from model.storage.local_model_store import LocalModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
from transformers import AutoModelForCausalLM

class ModelUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    def __init__(
        self,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        model_tracker: ModelTracker,
        comps: dict,
    ):
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker
        self.competitions = comps

    def set_competitions(self, comp):
        self.competitions = comp

    async def sync_model(self, hotkey: str, metadata, force: bool = False) -> bool:
        """Updates local model for a hotkey if out of sync and returns if it was updated.

        Args:
           hotkey (str): The hotkey of the model to sync.
           force (bool): Whether to force a sync for this model, even if it's chain metadata hasn't changed.
        """

        if self.competitions is None:
            bt.logging.debug("Competitions not known")
            return False

        if not metadata:
            return False

        if metadata.id.competition not in self.competitions:
            bt.logging.trace(
                f"Hotkey {hotkey} advertized model for invalid competition {metadata.id.competition}"
            )
            return False
        cname = metadata.id.competition
        cparams = self.competitions[cname]

        # Check what model id the model tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )

        # If we are not forcing a sync due to retrying a top model we can short-circuit if no change.
        if not force and metadata == tracker_model_metadata:
            return False

        # Get the local path based on the local store to download to (top level hotkey path)
        model_available = False
        snapshot_path = utils.get_local_model_snapshot_dir(self.local_store.base_dir, hotkey, metadata.id)
        if os.path.exists(snapshot_path):
            current_hash = utils.get_hash_of_directory(snapshot_path)
            hash_with_hotkey = get_hash_of_two_strings(current_hash, hotkey)
            if hash_with_hotkey == metadata.id.hash:
                model_available = True

        # Otherwise we need to download the new model based on the metadata.
        if model_available:
            bt.logging.debug(f"Model {metadata.id} already available, not downloading")
            pt_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=snapshot_path,
                use_safetensors=True,
            )
        else:
            model_size_limit = cparams.get('model_size', constants.MAX_MODEL_SIZE)
            statvfs = os.statvfs(self.local_store.base_dir)
            bytes_free = statvfs.f_bsize*statvfs.f_bavail
            bytes_free_required = model_size_limit + constants.DOWNLOAD_MIN_FREE_MARGIN_GB*1e9
            if bytes_free < bytes_free_required:
                gb_total = int(statvfs.f_frsize*statvfs.f_blocks//1e9)
                bt.logging.warning(f'not downloading model {metadata.id}: {bytes_free/1e9:.1f} of {gb_total} GB free while {bytes_free_required/1e9:.1f} GB required')
                return False
            path = self.local_store.get_path(hotkey)
            bt.logging.debug(f"download_model({metadata.id}) to {path}")
            try:
                model = await self.remote_store.download_model(
                    metadata.id, path, model_size_limit
                )
            except Exception as e:
                bt.logging.trace(
                    f"Failed to download model for hotkey {hotkey} due to {e}."
                )
                return False

            # Check that the hash of the downloaded content matches.
            hash_with_hotkey = get_hash_of_two_strings(model.id.hash, hotkey)
            if hash_with_hotkey != metadata.id.hash:
                bt.logging.trace(
                    f"Sync for hotkey {hotkey} failed. Hash of content downloaded from hugging face {model.id.hash} "
                    + f"or the hash including the hotkey {hash_with_hotkey} do not match chain metadata {metadata}."
                )
                return False
            pt_model = model.pt_model

        # Check that the model parameters are allowed in the proposed competition
        mdl_allowed, reason = competitions.validate_model_constraints(pt_model, cparams)
        if not mdl_allowed:
            bt.logging.trace(
                f"Sync for hotkey {hotkey} failed: model not allowed in competition {cname}: {reason}"
            )
            return False

        # Update the tracker
        self.model_tracker.on_miner_model_updated(hotkey, metadata)

        return True
