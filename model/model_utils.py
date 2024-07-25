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

import os
import sys
import time
import torch
import base64
import hashlib
from typing import Optional
import constants
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
import bittensor as bt
from transformers import PreTrainedModel, AutoModelForCausalLM
from safetensors.torch import load_model
from utilities import utils
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    GPT2TokenizerFast,
)

# 769_782_400 param model as a sample.
def get_test_model():
    config = GPTNeoXConfig(
        vocab_size=10000,
        num_attention_heads=40,
        hidden_size=1600,
        intermediate_size=6400,
        num_hidden_layers=24,
        max_position_embeddings=2048,
    )
    return GPTNeoXForCausalLM(config)


def get_hash_of_two_strings(string1: str, string2: str) -> str:
    """Hashes two strings together and returns the result."""

    string_hash = hashlib.sha256((string1 + string2).encode())

    return base64.b64encode(string_hash.digest()).decode("utf-8")


def model_path(base_dir: str, run_id: str) -> str:
    """
    Constructs a file path for storing the model relating to a training run.
    """
    return os.path.join(base_dir, "training", run_id)


async def push_model_id(
    model_id: ModelId,
    wallet: bt.wallet,
    retry_delay_secs: int = 60,
    metadata_store: Optional[ModelMetadataStore] = None,
    subtensor = None
):
    bt.logging.success(f"Now committing to the chain with model_id: {model_id}\n{model_id.to_compressed_str()}")

    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(subtensor, wallet)

    # We can only commit to the chain every 20 minutes, so run this in a loop, until
    # successful.
    for j in range(10):
        if j:
            bt.logging.error(f"Could not read back commitment, retrying from start (#{j}).")
        try:
            await metadata_store.store_model_metadata(wallet.hotkey.ss58_address, model_id)
            bt.logging.info("Wrote model metadata to the chain. Checking we can read it back...")
        except Exception as e:
            bt.logging.error(f"Commit maybe failed? {e}")

        for i in range(10):
            if i:
                bt.logging.error(f"Read-back failed, retrying read (#{i})")
            try:
                model_metadata = await metadata_store.retrieve_model_metadata(wallet.hotkey.ss58_address)

                if model_metadata and model_metadata.id == model_id:
                    bt.logging.success("Read back successful; committed model to the chain.")
                    return True
                bt.logging.error(f"Failed to read back model metadata from the chain. Expected: {model_id}, got: {model_metadata}. Retrying commit.")
                break
            except Exception as e:
                bt.logging.error(f"Exception reading metadata from chain: {e}")
            time.sleep(retry_delay_secs)

    return False


async def push(
    model: PreTrainedModel,
    repo: str,
    wallet: bt.wallet,
    retry_delay_secs: int = 60,
    metadata_store: Optional[ModelMetadataStore] = None,
    remote_model_store: Optional[RemoteModelStore] = None,
    competition = None,
    subtensor = None,
):
    """Pushes the model to Hugging Face and publishes it on the chain for evaluation by validators.

    Args:
        model (PreTrainedModel): The model to push.
        repo (str): The repo to push to. Must be in format "namespace/name".
        wallet (bt.wallet): The wallet of the Miner uploading the model.
        retry_delay_secs (int): The number of seconds to wait before retrying to push the model to the chain.
        metadata_store (Optional[ModelMetadataStore]): The metadata store. If None, defaults to writing to the
            chain.
        remote_model_store (Optional[RemoteModelStore]): The remote model store. If None, defaults to writing to HuggingFace
    """
    bt.logging.info("Pushing model")

    if remote_model_store is None:
        remote_model_store = HuggingFaceModelStore()

    if competition is None:
        competition = constants.COMPETITION_ID

    # First upload the model to HuggingFace.
    namespace, name = utils.validate_hf_repo_id(repo)
    model_id = ModelId(namespace=namespace, name=name)
    model_id = await remote_model_store.upload_model(Model(id=model_id, pt_model=model))

    bt.logging.success("Uploaded model to hugging face.")

    bt.logging.info(
        f"Hashing miner hotkey {wallet.hotkey.ss58_address} into the hash before uploading."
    )
    new_hash = get_hash_of_two_strings(model_id.hash, wallet.hotkey.ss58_address)
    model_id = model_id.copy(update={"hash": new_hash, "competition": competition})

    return await push_model_id(
            model_id=model_id,
            wallet=wallet,
            retry_delay_secs=retry_delay_secs,
            metadata_store=metadata_store,
            subtensor=subtensor
    )


def save(model: PreTrainedModel, model_dir: str):
    """Saves a model to the provided directory"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Save the model state to the specified path.
    model.save_pretrained(
        save_directory=model_dir,
        safe_serialization=True,
    )


async def get_repo(
    uid: int,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
) -> str:
    """Returns a URL to the HuggingFace repo of the Miner with the given UID."""
    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(bt.subtensor())
    if metagraph is None:
        metagraph = bt.metagraph(netuid=constants.SUBNET_UID)
    hotkey = metagraph.hotkeys[uid]
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)

    if not model_metadata:
        raise ValueError(f"No model metadata found for miner {uid}")

    return utils.get_hf_url(model_metadata)

def convert_dtype(dtype):
    if type(dtype) is torch.dtype:
        return dtype
    if dtype == 'bfloat16':
        return torch.bfloat16
    if dtype == 'float16':
        return torch.float16
    if dtype == 'float32':
        return torch.float32
    raise ValueError("Unknown torch datatype {dtype}")

def load_local_model(model_dir: str, dtype='bfloat16') -> PreTrainedModel:
    """Loads a model from a directory."""
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=convert_dtype(dtype)
    )

def best_uid(metagraph: Optional[bt.metagraph] = None) -> int:
    """Returns the best performing UID in the metagraph."""
    if not metagraph:
        metagraph = bt.subtensor().metagraph(constants.SUBNET_UID)
    return max(range(metagraph.n), key=lambda uid: metagraph.I[uid].item())


async def load_best_model(download_dir: str):
    """Loads the model from the best performing miner to download_dir"""
    best_uid = best_uid()
    return await load_remote_model(best_uid, download_dir)


async def load_remote_model(
    uid: int,
    download_dir: str,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
    remote_model_store: Optional[RemoteModelStore] = None,
) -> PreTrainedModel:
    """Loads the model currently being advertised by the Miner with the given UID.

    Args:
        uid (int): The UID of the Miner who's model should be downloaded.
        download_dir (str): The directory to download the model to.
        metagraph (Optional[bt.metagraph]): The metagraph of the subnet.
        metadata_store (Optional[ModelMetadataStore]): The metadata store. If None, defaults to reading from the
        remote_model_store (Optional[RemoteModelStore]): The remote model store. If None, defaults to reading from HuggingFace
    """

    if metagraph is None:
        metagraph = bt.metagraph(netuid=constants.SUBNET_UID)

    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(subtensor=bt.subtensor())

    if remote_model_store is None:
        remote_model_store = HuggingFaceModelStore()

    hotkey = metagraph.hotkeys[uid]
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)
    if not model_metadata:
        raise ValueError(f"No model metadata found for miner {uid}")

    bt.logging.success(f"Fetched model metadata: {model_metadata}")
    model: Model = await remote_model_store.download_model(
        model_metadata.id, download_dir
    )
    return model.pt_model
