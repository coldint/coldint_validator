# Miner
Miners are rewarded for improving the model that is currently the object of research.

The base miner provided by subnet 9 (Pretrain) has been left in the code-base, but should be seen as starting point only.
It is unlikely it will produce much improved models. We plan on releasing a more powerful training tool in the coming months.

Miners train locally and periodically publish their best model to hugging face and commit the metadata for that model to the Bittensor chain.

Miners can only have one model associated with them on the chain for evaluation by validators at a time. The list of allowed model types by block can be found in [constants/\_\_init\_\_.py](../constants/__init__.py).

The communication between a miner and a validator happens asynchronously chain and therefore Miners do not need to be running continuously. Validators will use whichever metadata was most recently published by the miner to know which model to download from hugging face.

# System Requirements

Miners will need enough disk space to store the model they work on. Max size of model is defined in [constants/\_\_init\_\_.py](../constants/__init__.py), but is typically 15GB. It is recommended to have at least 100 GB of disk space.

Miners will need enough processing power to train their model. The current models have around 7B parameters. To train such a model a single large GPU (80 GB) is required, or multiple 48GB or 24GB GPUs.

# Getting started

## Prerequisites

1. Get a Hugging Face Account: 

Miner and validators use hugging face in order to share model state information. Miners will be uploading to hugging face and therefore must attain a account from [hugging face](https://huggingface.co/) along with a user access token which can be found by following the instructions [here](https://huggingface.co/docs/hub/security-tokens).

Make sure that any repo you create for uploading is public so that the validators can download from it for evaluation.

2. Clone the repo, setup venv and install requirements

```shell
# Clone repo
git clone https://github.com/coldint/coldint_validator.git
cd coldint_validator

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

3. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

4. Produce a model, upload using the [upload script](../scripts/upload_model.py). The script requires access to your hugging face acount. This can be through a read/write token in your environment or in a .env file:
```shell
HF_ACCESS_TOKEN="YOUR_HF_ACCESS_TOKEN"
echo "HF_ACCESS_TOKEN=YOUR_HF_ACCESS_TOKEN" >.env
```
(or using ```huggingface-cli login```)

---

# Benchmarking a model
To benchmark the performance of a new model, it is recommended to start a validator.
Then create a benchmark.json file ([example](../benchmark_example.json)) to inject the
model into the evaluation loop and see how it performs.
