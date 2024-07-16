import datetime as dt
from pathlib import Path
from transformers import (
    GPTNeoXForCausalLM,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
)

# ---------------------------------
# Project Constants.
# ---------------------------------

# Release
__version__ = "0.9.0"

# Validator schema version
__validator_version__ = "0.9.0"
version_split = __validator_version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
weights_version_key = __spec_version__

# The validator WANDB config
WANDB_ENTITY    = "coldint"
WANDB_PROJECT   = "sn29"

# Subnet info
SUBNET_UID      = 29
SUBNET_N_UIDS   = 256

# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent

# Model parameters
MAX_MODEL_BYTES         = 15*1024*1024*1024
MAX_MODEL_PARAMETERS    = 6_900_000_000

ALLOWED_MODEL_TYPES = {
    GPTNeoXForCausalLM,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
}

# The number of run steps to log to single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100


# validator weight moving average term
weight_alpha = 0.5

blocks_per_epoch = 361
# Initial advantage of oldest model
advantage_initial = 0.004
# Advantage decay factor. 0.995 results in ~50% decay in one week
advantage_decay_per_epoch = 0.995

# validators number of pages to eval over miners on each step.`
n_eval_pages = 8
# validator eval batch size.
batch_size = 1
# validator eval batch min to keep for next loop.
sample_min = 6
# validator eval batch max. Difference from min is room to eval newly uploaded models.
sample_max = 14
# validator incentive threshold to prioritize updates. All incentives add up to 1.
update_priority_incentive_threshold = 0.01
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# time required between retrying evaluation of a stale model. (First retry will be immediate).
model_retry_cadence = dt.timedelta(hours=4)
