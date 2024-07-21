# Validator

Validators download the models from hugging face for each miner based on the Bittensor chain metadata and continuously evaluate them, setting weights based on the performance of each model against the Fineweb-2 dataset. They also log results to [wandb](https://wandb.ai/coldint/sn29).

You can view the entire validation system by reading the code in neurons/validator.py. SN29 has an improved scoring mechanism compared to the original pretraining subnet.

# System Requirements

Validators will need enough disk space to store the models of miners being evaluated. There is a maximum model size, currently ~15GB and 6.9B parameters, defined in [constants/\_\_init\_\_.py](../constants/__init__.py) and the validator has cleanup logic to remove old models. It is recommended to have at least 1 TB of disk space.

Validators will need enough processing power to evaluate their model, an RTX4090 (with 24GB RAM) is the minimum recommend GPU.

# Getting Started

## Prerequisites

1. Clone the repo, setup venv and install requirements

```shell
# Clone repo
git clone https://github.com/coldint/coldint_validator.git
cd condint_validator

# Setup venv
python -m venv coldint_venv
. coldint_venv/bin/activate

# Pre-install several packages, there are some dependency issues
pip install packaging
pip install wheel
pip install torch

# Install package including requirements
pip install -e .
```

2. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

3. (Optional) Run a Subtensor instance:
Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's.
We recommend running a local node as follows and passing the `--subtensor.network local` flag to your running miners/validators.
To install and run a local subtensor node follow the instructions on [the GitHub page of subtensor](https://github.com/opentensor/subtensor).

---

# Running the Validator

## With auto-updates

We highly recommend running the validator with auto-updates. This will help ensure your validator is always running the latest release, helping to maintain a high vtrust.

Prerequisites:
1. To run with auto-update, you will need to have [pm2](https://pm2.keymetrics.io/) installed.
2. Make sure your virtual environment is activated. This is important because the auto-updater will automatically update the package dependencies with pip.
3. Make sure you're using the main branch: `git checkout main`.

From the pretraining folder:
```shell
pm2 start --name net29-vali-updater --interpreter python scripts/start_validator.py -- --pm2_name net29-vali --wallet.name coldkey --wallet.hotkey hotkey [other vali flags]
```

This will start a process called `net29-vali-updater`. This process periodically checks for a new git commit on the current branch. When one is found, it performs a `pip install` for the latest packages, and restarts the validator process (who's name is given by the `--pm2\_name` flag)

## Without auto-updates

If you'd prefer to manage your own validator updates...

From the pretraining folder:
```shell
pm2 start python -- ./neurons/validator.py --wallet.name coldkey --wallet.hotkey hotkey
```

# Configuration

## Flags

The Validator offers some flags to customize properties, such as the device to evaluate on and the number of models to evaluate each step.

You can view the full set of flags by running
```shell
python ./neurons/validator.py -h
```

## Test Running Validation

Test running validation:
```shell
python neurons/validator.py
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_WALLET_HOTKEY
    --device YOUR_CUDA DEVICE
    --wandb.off
    --offline
```
---
