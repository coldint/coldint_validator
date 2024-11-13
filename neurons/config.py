import argparse
import os
import bittensor as bt
import torch
import constants


def validator_config():
    """Returns the config for the validator."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device name.",
    )
    parser.add_argument(
        "--wandb.off",
        dest="wandb.on",
        action="store_false",
        help="Turn off wandb logging.",
    )
    parser.add_argument(
        "--dont_set_weights",
        action="store_true",
        help="Validator does not set weights on the chain.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean models.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "model-store/"),
        help="Where to store downloaded models",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--save_step_json",
        type=str,
        help="Write step JSON file here",
    )
    parser.add_argument(
        "--model_store_size_gb",
        default=-constants.DEFAULT_MIN_FREE_GB,
        metavar='GB',
        type=int,
        help="Maximum size of model store (>0) or minimum space to keep free on disk (<=0) after model cleanup; please keep enough free space to download new models.",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config
