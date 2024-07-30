#!/usr/bin/env python

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
import sys
import time
import argparse
import constants
from model import model_utils
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import bittensor as bt
from utilities import utils
from model.data import ModelId

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"

args = None

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
    parser.add_argument('--dtype',
        default='bfloat16',
        choices=['bfloat16','float16','float32'],
        help='Convert model datatype before upload, bfloat16 is default'
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=constants.SUBNET_UID,
        help="The subnet UID (default 29 coldint)",
    )
    parser.add_argument(
        "--competition",
        type=str,
        default=constants.COMPETITION_ID,
        help="Competition to use in model id (defaults to c00)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Submit model id to chain instead of loading/uploading/downloading"
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    global args
    args = parser.parse_args()

    return config


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config)

    attempt = 0;
    while True:
        if attempt >= 3:
            return False
        attempt += 1
        try:
            wallet = bt.wallet(config=config)
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(config.netuid)

            bt.logging.success(f"Subtensor block: {subtensor.block}")
            # Make sure we're registered and have a HuggingFace token.
            utils.assert_registered(wallet, metagraph)
            if args.model_id is None:
                HuggingFaceModelStore.assert_access_token_exists()
            break

        except Exception as e:
            bt.logging.error(f"Failed to connect, retrying in 30sec: {e}")
            time.sleep(30)

    if args.model_id is None:
        # Load the model from disk and push it to the chain and Hugging Face.
        model = model_utils.load_local_model(config.load_model_dir,dtype=args.dtype)
        await model_utils.push(
            model=model,
            repo=config.hf_repo_id,
            wallet=wallet,
            competition=args.competition,
            subtensor=subtensor,
            private=True,
        )
        bt.logging.info("Model uploaded as private, please make public using scripts/change_repo_visibility.py!")
    else:
        parts = args.model_id.split(':')
        if len(parts) != 5:
            bt.logging.error(f"model_id format is user:repo:commithash:modelhash:competition")
            return False
        model_id = ModelId.from_compressed_str(args.model_id)
        if model_id.to_compressed_str() != args.model_id:
            bt.logging.error(f"model_id did not parse very well: {model_id.to_compressed_str()} != {args.model_id}")
            return False
        bt.logging.success(f"pushing {model_id.to_compressed_str()} to chain")
        bt.logging.success(f"Subtensor block: {subtensor.block}")
        result = await model_utils.push_model_id(
            model_id=model_id,
            wallet=wallet,
            subtensor=subtensor,
        )
        bt.logging.success(f"Result={result}")
        try:
            bt.logging.success(f"subtensor block: {subtensor.block}")
        except:
            bt.logging.success(f"failed to read subtensor block (no issue)")

    return True


def mmain():
    global config
    bt.logging.set_debug()
    bt.logging.set_trace()
    # Parse and print configuration
    config = get_config()
    if False:
        print(config)
        return False
    else:
        return asyncio.run(main(config))

if __name__ == "__main__":
    try:
        ret = mmain()
        sys.exit(0 if ret else -1)
    except KeyboardInterrupt:
        sys.exit(-1)
