import bittensor as bt
from typing import Optional
import constants
import os
from model import competitions
from model.model_utils import get_hash_of_two_strings
from model.data import ModelMetadata, ModelIssue, ModelLockedException
from model.storage.disk import utils
from model.storage.local_model_store import LocalModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
from transformers import AutoModelForCausalLM
import torch
import traceback

class ModelUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    SYNC_RESULT_SUCCESS             = 0
    SYNC_RESULT_ERROR               = -1
    SYNC_RESULT_DOWNLOAD_FAILED     = -2
    SYNC_RESULT_HASH_FAILED         = -3
    SYNC_RESULT_MODEL_NOT_ALLOWED   = -4
    RETRY_RESULTS = {
        SYNC_RESULT_ERROR,
        SYNC_RESULT_DOWNLOAD_FAILED,
    }

    def __init__(
        self,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        comps: dict,
    ):
        self.remote_store = remote_store
        self.local_store = local_store
        self.competitions = comps

    def set_competitions(self, comp):
        self.competitions = comp

    async def sync_model(self, hotkey: str, metadata) -> bool:
        """Updates local model for a hotkey if out of sync and returns if it was updated.

        Args:
           hotkey (str): The hotkey of the model to sync.
        """

        if self.competitions is None:
            bt.logging.debug("Competitions not known")
            return self.SYNC_RESULT_ERROR

        if not metadata:
            return self.SYNC_RESULT_ERROR

        if metadata.id.competition not in self.competitions:
            bt.logging.trace(
                f"Hotkey {hotkey} advertized model for invalid competition {metadata.id.competition}"
            )
            return self.SYNC_RESULT_ERROR
        cname = metadata.id.competition
        cparams = self.competitions[cname]

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
            with torch.device('meta'):
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
                return self.SYNC_RESULT_DOWNLOAD_FAILED
            path = self.local_store.get_path(hotkey)
            bt.logging.debug(f"download_model({metadata.id}) to {path}")
            try:
                with torch.device('meta'):
                    model = await self.remote_store.download_model(
                        metadata.id, path, model_size_limit
                    )
            except ModelIssue as e:
                # Indicates a reason the model is not allowed
                bt.logging.info(f"Model not allowed: {e}")
                return self.SYNC_RESULT_MODEL_NOT_ALLOWED
            except (OSError,ModelLockedException) as e:
                bt.logging.debug(
                    f"Failed to download model for hotkey {hotkey} due to {type(e).__name__} '{str(e).replace(os.linesep,' ')}'"
                )
                return self.SYNC_RESULT_DOWNLOAD_FAILED
            except Exception as e:
                # Include traceback for unanticipated exceptions.
                bt.logging.debug(
                    f"Failed to download model for hotkey {hotkey} due to {type(e).__name__} // {e}.\n{traceback.format_exc()}"
                )
                return self.SYNC_RESULT_DOWNLOAD_FAILED

            # Check that the hash of the downloaded content matches.
            hash_with_hotkey = get_hash_of_two_strings(model.id.hash, hotkey)
            if hash_with_hotkey != metadata.id.hash:
                bt.logging.trace(
                    f"Sync for hotkey {hotkey} failed. Hash of hugging face content and hotkey {hash_with_hotkey} does not match chain metadata {metadata}."
                )
                return self.SYNC_RESULT_HASH_FAILED
            pt_model = model.pt_model

        # Check that the model parameters are allowed in the proposed competition
        mdl_allowed, reason = competitions.validate_model_constraints(pt_model, cparams)
        if not mdl_allowed:
            bt.logging.trace(
                f"Sync for hotkey {hotkey} failed: model not allowed in competition {cname}: {reason}"
            )
            return self.SYNC_RESULT_MODEL_NOT_ALLOWED

        return self.SYNC_RESULT_SUCCESS
