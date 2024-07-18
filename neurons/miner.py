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

# Example miner / model training code.
# Do not expect much without extensive tuning of parameters or code updates!
# (Parameters depend heavily on the state of the model you're starting from)

import asyncio
import math
import os
import wandb
import torch
import random
import argparse
import constants
import dataset
from model import model_utils
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
import bittensor as bt
from transformers import PreTrainedModel
from utilities import utils
import datetime as dt

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.

    Returns:
        A namespace object containing the configuration parameters.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not send model to wandb, does not check if registered",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="The wandb project to log to."
    )
    parser.add_argument("--wandb_entity", type=str, help="The wandb entity to log to.")
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "local-models/"),
        help="Where to download/save models for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device on which to run. cpu or cuda",
    )
    parser.add_argument(
        "--load_best",
        action="store_true",
        help="If set, the miner loads the best model from wandb to train off.",
    )
    parser.add_argument(
        "--load_uid",
        type=int,
        default=None,
        help="If passed loads the model under the specified uid.",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified directory",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="If provided, loads the safetensor serialized model from the specified file."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=-1,
        help="Number of training epochs (-1 is infinite)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save model after this many epochs",
    )
    parser.add_argument("--pack-samples", default=False, action="store_true", help="Pack samples")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wdecay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument(
        "--bs", type=int, default=constants.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--sl", type=int, default=2048, help="(Max) sequence length"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=5,
        help="The number of training accumulation steps.",
    )
    parser.add_argument(
        "--pages_per_epoch",
        type=int,
        default=10,
        help="Number of pages trained on per epoch",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config


async def load_starting_model(
    config: bt.config,
    metagraph: bt.metagraph,
    metadata_store: ModelMetadataStore,
    remote_model_store: RemoteModelStore,
) -> PreTrainedModel:
    """Loads the model to train based on the provided config."""

    # Initialize the model based on the best on the network.
    if config.load_best:
        # Get the best UID be incentive and load it.
        best_uid = model.best_uid(metagraph)
        model = await model.load_remote_model(
            best_uid,
            config.model_dir,
            metagraph,
            metadata_store,
            remote_model_store,
        )
        bt.logging.success(
            f"Training with model from best uid: {best_uid}. Model={str(model)}"
        )
        return model

    # Initialize the model based on a passed uid.
    if config.load_uid is not None:
        # Sync the state from the passed uid.
        model = await model.load_remote_model(
            config.load_uid,
            config.model_dir,
            metagraph,
            metadata_store,
            remote_model_store,
        )
        bt.logging.success(
            f"Training with model from uid: {config.load_uid}. Model={str(model)}"
        )
        return model

    # Check if we should load a model from a local directory.
    if config.load_model_dir:
        model = model.load_local_model(config.load_model_dir)
        bt.logging.success(f"Training with model from disk. Model={str(model)}")
        return model

    # Check if we should load a model from a local file.
    if config.load_model:
        model = model.load_gpt2_model(config.load_model)
        bt.logging.success(f"Training with model from disk. Model={str(model)}")
        return model

    bt.logging.error("Please provide a starting model")
    return None


async def main(config: bt.config):
    bt.logging(config=config)

    # Create bittensor objects if interaction with the chain is required
    # (no need to be registered)
    wallet = subtensor = metagraph = remote_store = None
    if config.load_uid or config.load_best:
        subtensor = bt.subtensor(config=config)
        remote_store = HuggingFaceModelStore()

    # Create a unique run id for this run.
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = model_utils.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)

    use_wandb = False
    if not config.offline:
        if config.wandb_project is None or config.wandb_entity is None:
            bt.logging.warning(
                "Wandb project or entity not specified. This run will not be logged to wandb"
            )
        else:
            use_wandb = True

    # Init model.
    metadata_store = ChainModelMetadataStore(subtensor, None, config.netuid)
    model: PreTrainedModel = await load_starting_model(
        config, metagraph, metadata_store, remote_store
    )
    if model is None:
        return False
    model = model.train()
    model = model.to(config.device)

    bt.logging.success(f"Saving model to path: {model_dir}.")
    model.save(model, model_dir)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wdecay)
    wandb_run = None

    # If using wandb, start a new run.
    if use_wandb:
        token = os.getenv("WANDB_API_KEY")
        if not token:
            raise ValueError(
                "To use Wandb, you must set WANDB_API_KEY in your .env file"
            )

        wandb.login(key=token)

        wandb_run = wandb.init(
            name=run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={
                "run_name": run_id,
                "version": constants.__validator_version__,
                "type": "miner",
            },
            allow_val_change=True,
        )

        # At the end of the run, upload the model to wandb, for debugging purposes only.
        # This is not seen by validators.
        wandb_run.save(os.path.join(model_dir, "*"), base_path=model_dir, policy="end")
    else:
        bt.logging.warning(
            "Not posting run to wandb. Either --offline is specified or the wandb settings are missing."
        )

    # Start the training loop
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    accumulation_steps = config.accumulation_steps
    tokenizer = model_utils.get_tokenizer()

    try:
        while epoch_step < config.num_epochs or config.num_epochs == -1:
            # Initialize loss accumulator for the epoch
            epoch_loss = 0.0

            # Prepare the data loader with random pages for each epoch
            bt.logging.success(
                f"Loading {config.pages_per_epoch} pages for training this epoch"
            )
            loader = dataset.SubsetFineWebEdu2Loader(
                batch_size=config.bs,
                sequence_length=config.sl,
                num_pages=config.pages_per_epoch,
                tokenizer=tokenizer,
                pack=config.pack_samples,
            )

            # Enumerate over the data loader
            n_batches = 0
            optimizer.zero_grad()  # Initialize gradients to zero

            for i, batch in enumerate(loader):
                # Move the input batch to the device
                inputs = batch.to(model.device)

                # Forward pass: compute the model output and loss
                outputs = model(inputs, labels=inputs)

                loss = outputs.loss / accumulation_steps  # Scale loss
                loss.backward()  # Accumulate gradients

                if (i + 1) % accumulation_steps == 0:
                    n_acc_steps += 1
                    optimizer.step()  # Perform a single optimization step
                    optimizer.zero_grad()  # Clear gradients
                    bt.logging.success(
                        f"Step: {n_acc_steps} loss: {outputs.loss.detach().item()}"
                    )
                    if use_wandb:
                        wandb_run.log(
                            {"loss": outputs.loss.detach(), "n_batches": n_batches},
                            step=n_acc_steps,
                        )

                torch.cuda.empty_cache()

                n_batches += 1
                global_step += 1
                epoch_loss += outputs.loss.detach().item()

            # Calculate the average loss for the epoch
            avg_loss = epoch_loss / n_batches

            # Log the average loss for the epoch
            bt.logging.success(f"Epoch: {epoch_step} average loss: {avg_loss}")
            epoch_step += 1

            if (epoch_step % config.save_interval) == 0:
                bt.logging.success(f"Saving model to path: {model_dir}.")
                model.save(model, model_dir)

        bt.logging.success(f"Finished training, saving model to {model_dir}")
        model.save(model, model_dir)

    finally:
        # Important step.
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    print(config)

    asyncio.run(main(config))
