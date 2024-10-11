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
import random
import itertools
from utilities.mathutils import *

def group_samples(losses_per_uid, group_size):
    uids = list(losses_per_uid.keys())
    loss_mat = np.array([
        losses_per_uid[uid] for uid in uids
    ])

    # Zero samples where everyone scored NaN
    nans = np.isnan(loss_mat) | np.isinf(loss_mat)
    all_nan = np.sum(nans, axis=0) == loss_mat.shape[0]
    loss_mat[:, all_nan] = 0

    # Shuffle samples to reduce effect of correlations
    n_samples = loss_mat.shape[1]
    n_samples -= n_samples%group_size
    idxs = np.arange(n_samples)
    random.shuffle(idxs)
    loss_mat = loss_mat[:,idxs]

    # Note that a single NaN in the set results in NaN for the group
    summed = loss_mat.reshape(len(uids),-1,group_size).sum(axis=2)
    return {uid:np.ascontiguousarray(summed[i]) for i,uid in enumerate(uids)}

def compute_wins(
    losses_per_uid: typing.Dict[int, typing.List[float]],
    uid_to_block: typing.Dict[int, int],
    current_block,
    advantage_initial,
    advantage_decay_per_epoch
) -> typing.Dict[int, float]:
    """
    Computes win fractions.

    Parameters:
        losses_per_uid (dict): A dictionary of sample losses for each uid.
        uid_to_block (dict): A dictionary of blocks for each uid.
        current_block (int): current block id
        advantage_initial (float)
        advantage_decay_per_epoch (float)
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
            advantage_decay = advantage_decay_per_epoch**delta_epochs
            advantage_factor = 1 - advantage_initial * advantage_decay
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
            if won_all and not np.isnan(uid_a_loss) and not np.isinf(uid_a_loss):
                wins[uid_a] += 1
                break

    win_fractions = {uid: n_wins / n_samples for uid,n_wins in wins.items()}

    # Wins matrix (informative only)
    matrix = {uid_a:{uid_b:None for uid_b in uids_sorted} for uid_a in uids_sorted}
    for uid_a, uid_b in itertools.product(uids_sorted, uids_sorted):
        matrix[uid_a][uid_b] = {
            'loss': naninf_meandelta(losses_per_uid[uid_a],losses_per_uid[uid_b]),
            'wins': int(np.sum(losses_per_uid[uid_a] < losses_per_uid[uid_b])),
            'wins_adv': int(np.sum(uid_advantage_factors[uid_a] * losses_per_uid[uid_a] < losses_per_uid[uid_b])),
        }

    return dict(wins=wins, win_rate=win_fractions, advantage_factors=uid_advantage_factors, matrix=matrix)


def compute_losses_sliced(
    model, batches: typing.List[torch.Tensor], device: str, n_slices=1
) -> typing.List[float]:
    with torch.no_grad():
        sliced = model.sliced(
            n_slices=n_slices,
            device=device,
        )
        losses = sliced.evaluate_samples(batches,reduction='sum')
        bt.logging.info(f'computed sliced losses: {losses[:10]}...')
        return losses

def compute_losses_regular(
    model, batches: typing.List[torch.Tensor], device: str
) -> typing.List[float]:
    model.to(device)
    model.eval()

    losses = [math.inf]*len(batches) # Use infinity to indicate failure
    with torch.no_grad():
        cuda_errors = 0
        for i,batch in enumerate(batches):
            # None indicates the token sequence was too long or did not map back onto itself
            if batch is None:
                continue

            inputs = None
            logits = None
            try:
                inputs = batch.to(device)
                logits = model(inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                losses[i] = loss_fct(shift_logits, shift_labels).item()
            except Exception as e:
                bt.logging.error(f"Exception occurred: {e}")
                bt.logging.error(traceback.format_exc())
                if 'CUDA error' in str(e):
                    cuda_errors += 1
                    if cuda_errors>=4:
                        bt.logging.error(f'{cuda_errors} CUDA errors, bailing out of evaluation loop')
                        break
            del inputs
            del logits

    bt.logging.info(f'computed losses: {losses[:10]}...')

    return losses

def compute_losses(
    model, allow_sliced: bool, batches: typing.List[torch.Tensor], device: str
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
    bt.logging.info(f"Evaluating model type {type(model).__name__}")
    bt.logging.debug(f"Model: {model}")

    # Set to non-zero to compare sliced vs regular loss calculation
    test_sliced_eval = None

    regular_losses = None
    sliced_losses = None
    n_slices = test_sliced_eval
    if allow_sliced and hasattr(model,'sliced'):
        model_bytes = model.num_parameters()*model.dtype.itemsize
        gpu_ram = torch.cuda.get_device_properties(device).total_memory
        # The fraction below is somewhat arbitrary. The precise amount of ram
        # needed depends on many factors. TODO.
        arbitrary_fraction = 0.65
        use_gpu_ram = int(arbitrary_fraction * gpu_ram)
        if model_bytes > use_gpu_ram:
            # This assumes all slices are created equal, which isn't true.
            n_slices = (model_bytes+use_gpu_ram)//use_gpu_ram

    if n_slices is None or test_sliced_eval:
        regular_losses = compute_losses_regular(model,batches,device)

    if n_slices is not None:
        bt.logging.info(f"Performing {n_slices}-sliced eval: model ({model_bytes}) > {arbitrary_fraction} * gpu ram ({gpu_ram})")
        sliced_losses = compute_losses_sliced(model,batches,device,n_slices=n_slices)

    if regular_losses and sliced_losses:
        equal = sliced_losses==regular_losses
        nanequal = naninf_equal(sliced_losses,regular_losses)
        nanclose = naninf_close(sliced_losses,regular_losses)
        bt.logging.info(f'sliced losses == regular losses: {equal} / {nanequal} / {nanclose}')

    if regular_losses:
        return regular_losses

    return sliced_losses
