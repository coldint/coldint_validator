from pathlib import Path

# ---------------------------------
# Project Constants.
# ---------------------------------

# Release
__version__ = "1.0.6"

# Validator schema version
__validator_version__ = "1.0.6"
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

MAX_MODEL_SIZE          = 15*1024*1024*1024

# The number of run steps to log to single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100

# Hall of fame / competitions configuration
CFG_FETCH_INTERVAL      = 15*60
HOF_URL                 = "https://github.com/coldint/sn29/raw/main/hall_of_fame.json"
COMPETITIONS_URL        = "https://github.com/coldint/sn29/raw/main/competitions.json"

# Maximum fraction of (miner) emissions for rewards
REWARDS_MAX_FRACTION    = 0.4

# Bounty decay factor, per epoch
REWARDS_DECAY_FACTOR    = 0.995

# Infinite sum of emissions will be initial value times this factor (200 for 0.995)
# so: initial value equals total reward divided by REWARDS_IV_FACTOR
REWARDS_IV_FACTOR       = 1 / (1 - REWARDS_DECAY_FACTOR)

# Weights are deduced from win rates, by raising to this power and normalizing
WEIGHT_SKEW_FACTOR      = 1.2

# Fraction of weight to be considered a top miner on other validator (and re-evaluated once in a while)
TOP_MINER_FRACTION      = 0.1

MAX_SEQUENCE_LEN        = 4096
CAP_SAMPLE_LEN          = 20480
MAX_TOKENIZE_FAILS      = 3
TTL_RUN_STEP            = 7200
TTL_MODEL_EVAL          = 600

# validator weight moving average term
weight_alpha = 0.5

blocks_per_epoch = 361
# Initial advantage of oldest model
advantage_initial = 0.004
# Advantage decay factor. 0.995 results in ~50% decay in one week
advantage_decay_per_epoch = 0.995

# validators number of pages to eval over miners on each step.`
n_eval_pages = 8
n_rows_per_page = 100

DEFAULT_POOL_SIZE = 6
DEFAULT_PEND_SIZE = 6

CHAIN_UPDATE_INTERVAL       = 16*60
TOP_MODEL_RETRY_INTERVAL    = 4*3600
GEN_MODEL_RETRY_INTERVAL    = 24*3600
MODEL_RETRY_MAX_N_PER_LOOP  = 10
MODEL_UNFORCED_N_PER_LOOP   = 5
WEIGHT_SET_MIN_INTERVAL     = 25*60
LIMIT_MIN_FREE_GB           = 30
DEFAULT_MIN_FREE_GB         = 120
DOWNLOAD_MIN_FREE_MARGIN_GB = 5
MAX_VALIDATOR_AGE_BLOCKS    = (3600//12 * 4)
