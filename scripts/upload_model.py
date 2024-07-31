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
import huggingface_hub
import transformers
from model import model_utils
from model import competitions
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
        default=None,
        help="Competition to use in model id (auto-detect if not specified)"
    )
    parser.add_argument(
        "--check_competition",
        default=False, action='store_true',
        help="Check if model allowed in competition"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Submit model id to chain instead of loading/uploading/downloading (repo is not touched)"
    )
    parser.add_argument(
        "--keep_private",
        default=False, action='store_true',
        help="Keep model private after uploading"
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

    if args.load_model_dir is None and args.model_id is None:
        bt.logging.error(f"--load_model_dir or --model_id required")
        return -1

    if args.load_model_dir and args.model_id:
        bt.logging.warning("--load_moder_dir ignored when specifying --model_id")
    elif args.load_model_dir and args.hf_repo_id is None:
        bt.logging.error("--load_moder_dir requires --hf_repo_id to upload")
        return -1

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

        if args.competition is None or args.check_competition:
            compts = competitions.load_competitions(constants.COMPETITIONS_URL)
        if args.competition is None:
            valid_cs = competitions.model_get_valid_competitions(model, compts)
            if len(valid_cs) != 1:
                bt.logging.warning(f'Unable to determine competition, options: {valid_cs}')
                return -1
            bt.logging.info(f"Detected model only valid in competition {valid_cs[0]}, selecting")
            args.competition = valid_cs[0]
        if args.check_competition:
            valid, reason = competitions.validate_model_constraints(model, compts[args.competition])
            if not valid:
                bt.logging.warning(f'Model not valid in {args.competition}, reason: {reason}')
                return -1
            bt.logging.warning(f'Model valid in competition {args.competition}')

        # First make repo and set to private; otherwise existing repos will remain public
        huggingface_hub.create_repo(config.hf_repo_id, private=True, exist_ok=True)
        huggingface_hub.update_repo_visibility(config.hf_repo_id, private=True)

        tokenizer = None
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.load_model_dir)
            bt.logging.info(f"Including tokenizer {tokenizer}")
        except:
            bt.logging.info("No tokenizer found")

        await model_utils.push(
            model=model,
            repo=config.hf_repo_id,
            wallet=wallet,
            competition=args.competition,
            subtensor=subtensor,
            tokenizer=tokenizer,
            private=True,
        )
        bt.logging.info("Model pushed successfully")
        if config.keep_private:
            bt.logging.warning("Model uploaded as private, please make public using scripts/change_repo_visibility.py!")
        else:
            bt.logging.info("Setting repo to public...")
            huggingface_hub.update_repo_visibility(config.hf_repo_id, private=False)
            bt.logging.info("Repo is now public")
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
