<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <source srcset="./assets/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

# **Bittensor Pretrain Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

[Leaderboard](https://huggingface.co/spaces/macrocosm-os/pretraining-leaderboard) • [Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/subnets/netuid-9/) • [Research](https://bittensor.com/whitepaper) 
</div>

---

# Introduction

The following documentation assumes you are familiar with basic Bittensor concepts: Miners, Validators, and incentives. If you need a primer, please check out https://docs.bittensor.com/learn/bittensor-building-blocks.

Bittensor subnet 9 rewards miners for producing pretrained Foundation-Models on the Falcon Refined Web dataset. It acts like a continuous benchmark whereby miners are rewarded for attaining the best losses on randomly sampled pages of Falcon given a consistent model architecture. The reward mechanism works as follows:

    1. Miners train and periodically publish models to hugging face and commit the metadata for that model to the Bittensor chain.
    2. Validators download the models from hugging face for each miner based on the Bittensor chain metadata and continuously evaluate them, setting weights based on the performance of each model against the Falcon dataset. They also log results to [wandb](https://wandb.ai/opentensor-dev/pretraining-subnet).
    3. The Bittensor chain aggregates weights from all active validators using Yuma Consensus to determine the proportion of TAO emission rewarded to miners and validators. 

See the [Miner](docs/miner.md) and [Validator](docs/validator.md) docs for more information about how they work, as well as setup instructions.

---

## Incentive Mechanism

Bittensor hosts multiple incentive mechanism through which miners are evaluated by validators for performing actions well. Validators perform the process of evaluation and 'set weights', which are transactions into Bittensor's blockchain. Each incentive mechanism in Bittensor is called a 'subnet' and has an identifier (This particular mechanism has subnet uid 9). Weights and the amount of TAO held by the validators become inputs to Bittensor's consensus mechanism called Yuma Consensus. YC drives validators towards a consensus, agreement about the value of the work done by miners. The miners with the highest agreed upon scores are minted TAO, the network digital currency.

Miners within this subnet are evaluated based on the number of times the model they have hosted has a lower loss than another model on the network when randomly sampling from the near infinite Falcon Refined Web pretraining dataset. To perform well, miners must attain the lowest loss on the largest number of random batches. Finding the best model and delta at the earliest block ensures the most incentive.

---

## Getting Started

TL;DR:
1. [Chat](https://discord.gg/bittensor)
2. [Leaderboard](https://huggingface.co/spaces/macrocosm-os/pretraining-leaderboard)

This repo's main conversation is carried out in the Bittensor [Discord](https://discord.gg/bittensor). Visit the 'pretraining' channel to ask questions and get real time feedback. You can view the ongoing running of the incentive mechanism, the best miners (see 'incentive'), the most in consensus validators (see 'vtrust') using this [taostats link](https://taostats.io/subnets/netuid-9/). The table shows all 256 participant UIDs with corresponding YC stats and earnings. 

See [Miner Setup](docs/miner.md#getting-started) to learn how to set up a Miner.

See [Validator Setup](docs/validator.md#getting-started) to learn how to set up a Validator.

---

## Feedback

We welcome feedback!

If you have a suggestion, please reach out on the Discord channel, or file an Issue.

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
```
