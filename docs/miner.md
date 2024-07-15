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

2. Clone the repo

```shell
git clone https://github.com/macrocosm-os/pretraining.git
```

3. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

4. Install the requirements. From your virtual environment, run
```shell
cd pretraining
python -m pip install -e .
```

Note: flash-attn may not have their dependencies set up correctly. If you run into issues try installing those requirements separately first:
```shell
pip install packaging
pip install wheel
pip install torch
```

5. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).
---

# Running the Miner

The mining script uploads a model to hugging face which will be evaluated by validators.

To evaluate your training results the recommended approach is to install the validator and run it with your models injected using the file `benchmark.json`. This allows you to see what would happen in a validator after publishing your model. It also gives valuable insights into how other models are performing.

## Env File

The Miner requires a .env file with your hugging face access token in order to upload models.

Create a `.env` file in the `coldint\_validator` directory and add the following to it:
```shell
HF_ACCESS_TOKEN="YOUR_HF_ACCESS_TOKEN"
```

## Starting the Miner

To start your miner the most basic command is

```shell
python neurons/miner.py --wallet.name coldkey --wallet.hotkey hotkey
```

- `--wallet.name`: should be the name of the coldkey that contains the hotkey your miner is registered with.

- `--wallet.hotkey`: should be the name of the hotkey that your miner is registered with.

### Flags

The Miner offers some flags to customize properties, such as how to train the model and which hugging face repo to upload to.

You can view the full set of flags by running
```shell
python ./neurons/miner.py -h
```

Some flags you may find useful:

- `--offline`: when set you can run the miner without being registered and it will not attempt to upload the model.

- `--wandb_entity` + `--wandb_project`: when both flags are set the miner will log its training to the provided wandb project.

- `--device`: by default the miner will use your gpu but you can specify with this flag if you have multiple.

#### Training from pre-existing models

- `--load_best`: when set you will download and train the model from the current best miner on the network.
- `--load_uid`: when passing a uid you will download and train the model from the matching miner on the network.
- `--load_model_dir`: the path to a local model directory [saved via Hugging Face API].
- `--load_model`: the path to a safetensors file [not necessarily saved from Hugging Face API].

---

## Manually uploading a model

In some cases you may have failed to upload a model or wish to upload a model without further training.

Due to rate limiting by the Bittensor chain you may only upload a model every 360 blocks (20 minutes).

You can manually upload with the following command:
```shell
python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey
```

## Running a custom miner

The list of allowed model types can be found in [constants/\_\_init\_\_.py](../constants/__init__.py)

In that file are also the constraints for the total number of parameters and the total size of the model.
