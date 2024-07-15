"""A script that pushes a model from disk to the subnet for evaluation.

Usage:
    python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey
    
Prerequisites:
   1. HF_ACCESS_TOKEN is set in the environment or .env file.
   2. load_model_dir points to a directory containing a previously trained model, with relevant Hugging Face files (e.g. config.json).
   3. Your miner is registered
"""

import asyncio
import os
import argparse
import constants
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import pretrain as pt
import bittensor as bt
from utilities import utils

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_config():
    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/pretraining",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified directory",
    )
    parser.add_argument(
        "--upload_b16",
        action="store_false",  # Defaults to True.
        help="If true, upload the model using bfloat16.",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--use_hotkey_in_hash",
        action="store_true",  # Defaults to False.
        help="If true, use the hotkey of the miner when generating the hash.",
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

    # Make sure we're registered and have a HuggingFace token.
    utils.assert_registered(wallet, metagraph)
    HuggingFaceModelStore.assert_access_token_exists()

    # Load the model from disk and push it to the chain and Hugging Face.
    model = pt.mining.load_local_model(config.load_model_dir, config.upload_b16)
    await pt.mining.push(
        model, config.hf_repo_id, wallet, use_hotkey_in_hash=config.use_hotkey_in_hash
    )


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    print(config)

    asyncio.run(main(config))
