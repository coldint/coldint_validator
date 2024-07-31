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

# Tools for performing validation over models.

import math
import torch
import typing
import constants
import traceback
import bittensor as bt
import numpy as np
import itertools

def compute_wins(
    losses_per_uid: typing.Dict[int, typing.List[float]],
    uid_to_block: typing.Dict[int, int],
    current_block
) -> typing.Dict[int, float]:
    """
    Computes win fractions.

    Parameters:
        losses_per_uid (dict): A dictionary of sample losses for each uid.
        uid_to_block (dict): A dictionary of blocks for each uid.
        current_block (int): current block id
    Returns:
        dictionary: computed dictionaries uid->value
                        wins
                        win_fractions
                        advantage_factors
                        matrix {uid_a -> {uid_b -> info}}
    """
    uids_sorted = sorted(losses_per_uid.keys(), key=lambda x: uid_to_block.get(x))
    if len(uids_sorted) == 0:
        return {}
    wins = {uid: 0 for uid in uids_sorted}
    losses_per_uid = {uid: np.array(losses) for uid, losses in losses_per_uid.items()}

    # Determine advantage factors
    uid_advantage_factors = {}
    for uid in uids_sorted:
        delta_blocks = current_block - uid_to_block.get(uid, 1<<31)
        delta_epochs = delta_blocks / constants.blocks_per_epoch
        if delta_epochs < 0 or math.isinf(delta_epochs):
            # model from after current_block (through benchmark)
            advantage_factor = 1
        else:
            # model from before current_block
            advantage_decay = constants.advantage_decay_per_epoch**delta_epochs
            advantage_factor = 1 - constants.advantage_initial * advantage_decay
        uid_advantage_factors[uid] = advantage_factor

    # For each sample, determine winner and award 1 point
    n_samples = len(losses_per_uid[uids_sorted[0]])
    for sample_idx in range(n_samples):
        sample_loss = [losses_per_uid[uid][sample_idx] for uid in uids_sorted]

        for i_uid_a, uid_a in enumerate(uids_sorted):
            # uid_a should win from all other models using its advantage factor
            uid_a_loss = sample_loss[i_uid_a] * uid_advantage_factors[uid_a]
            won_all = True
            for i_uid_b in range(i_uid_a+1,len(uids_sorted)):
                if uid_a_loss > sample_loss[i_uid_b]:
                    won_all = False
                    break
            if won_all:
                wins[uid_a] += 1
                break

    win_fractions = {uid: n_wins / n_samples for uid,n_wins in wins.items()}

    # Wins matrix (informative only)
    matrix = {uid_a:{uid_b:None for uid_b in uids_sorted} for uid_a in uids_sorted}
    for uid_a, uid_b in itertools.product(uids_sorted, uids_sorted):
        matrix[uid_a][uid_b] = {
            'loss': np.mean(losses_per_uid[uid_a] - losses_per_uid[uid_b]),
            'wins': np.sum(losses_per_uid[uid_a] < losses_per_uid[uid_b]),
            'wins_adv': np.sum(uid_advantage_factors[uid_a] * losses_per_uid[uid_a] < losses_per_uid[uid_b]),
        }

    return dict(wins=wins, win_rate=win_fractions, advantage_factors=uid_advantage_factors, matrix=matrix)


def compute_losses(
    model, batches: typing.List[torch.Tensor], device: str
) -> typing.List[float]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (dict): A list of batches.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        list: A list of losses for each batch.
    """
    bt.logging.info(f"Evaluating {model}")
    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():

        for batch in batches:
            # None indicates the token sequence was too long or did not map back onto itself
            if batch is None:
                losses.append(math.inf)
                continue

            try:
                inputs = batch.to(device)
                logits = model(inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                loss = loss_fct(shift_logits, shift_labels).item()

                losses.append(loss)
            except Exception as e:
                bt.logging.error(f"Exception occurred: {e}")
                traceback.print_exc()
                losses.append(math.inf)  # Use infinity to indicate failure
            del inputs
            del logits

    return losses
