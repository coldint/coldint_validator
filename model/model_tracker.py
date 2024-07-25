import copy
import threading
from typing import Dict, List, Optional, Set
import pickle
import bittensor as bt

from model.data import ModelMetadata


class ModelTracker:
    """Tracks the current model for each miner.

    Thread safe.
    """

    def __init__(
        self,
    ):
        # Create a dict from miner hotkey to model metadata.
        self.miner_hotkey_to_model_metadata_dict = dict()

        # Make this class thread safe because it will be accessed by multiple threads.
        # One for the downloading new models loop and one for the validating models loop.
        self.lock = threading.RLock()

    def get_state(self):
        with self.lock:
            return {
                hotkey: meta.dict()
                    for hotkey, meta in self.miner_hotkey_to_model_metadata_dict.items()
            }

    def set_state(self, state):
        with self.lock:
            self.miner_hotkey_to_model_metadata_dict = {
                hotkey: ModelMetadata(**info)
                    for hotkey, info in state.items()
            }

    def get_miner_hotkey_to_model_metadata_dict(self) -> Dict[str, ModelMetadata]:
        """Returns the mapping from miner hotkey to model metadata."""

        # Return a copy to ensure outside code can't modify the scores.
        with self.lock:
            return copy.deepcopy(self.miner_hotkey_to_model_metadata_dict)

    def get_model_metadata_for_miner_hotkey(
        self, hotkey: str
    ) -> Optional[ModelMetadata]:
        """Returns the model metadata for a given hotkey if any."""

        with self.lock:
            if hotkey in self.miner_hotkey_to_model_metadata_dict:
                return self.miner_hotkey_to_model_metadata_dict[hotkey]
            return None

    def on_hotkeys_updated(self, incoming_hotkeys: Set[str]):
        """Notifies the tracker which hotkeys are currently being tracked on the metagraph."""

        with self.lock:
            existing_hotkeys = set(self.miner_hotkey_to_model_metadata_dict.keys())
            for hotkey in existing_hotkeys - incoming_hotkeys:
                del self.miner_hotkey_to_model_metadata_dict[hotkey]
                bt.logging.trace(f"Removed outdated hotkey: {hotkey} from ModelTracker")

    def on_miner_model_updated(
        self,
        hotkey: str,
        model_metadata: ModelMetadata,
    ) -> None:
        """Notifies the tracker that a miner has had their associated model updated.

        Args:
            hotkey (str): The miner's hotkey.
            model_metadata (ModelMetadata): The latest model metadata of the miner.
        """
        with self.lock:
            self.miner_hotkey_to_model_metadata_dict[hotkey] = model_metadata

            bt.logging.trace(f"Updated Miner {hotkey}. ModelMetadata={model_metadata}.")
