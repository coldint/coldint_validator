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


def iswin(loss_i, loss_j, block_i, block_j) -> bool:
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust loss based on timestamp and pretrain epsilon
    loss_i = (1 - constants.timestamp_epsilon) * loss_i if block_i < block_j else loss_i
    loss_j = (1 - constants.timestamp_epsilon) * loss_j if block_j < block_i else loss_j
    return loss_i < loss_j


def compute_wins(
    uids: typing.List[int],
    losses_per_uid: typing.Dict[int, typing.List[float]],
    batches: typing.List[torch.FloatTensor],
    uid_to_block: typing.Dict[int, int],
) -> typing.Tuple[typing.Dict[int, int], typing.Dict[int, float]]:
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        losses_per_uid (dict): A dictionary of losses for each uid by batch.
        batches (List): A list of data batches.
        uid_to_block (dict): A dictionary of blocks for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = uid_to_block[uid_i]
        for j, uid_j in enumerate(uids):
            if i == j:
                continue
            block_j = uid_to_block[uid_j]
            for batch_idx, _ in enumerate(batches):
                loss_i = losses_per_uid[uid_i][batch_idx]
                loss_j = losses_per_uid[uid_j][batch_idx]
                wins[uid_i] += 1 if iswin(loss_i, loss_j, block_i, block_j) else 0
                total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    return wins, win_rate


def check_for_reasonable_output(
    model, input1: torch.Tensor, input2: torch.Tensor, pad_token_id: int
) -> bool:
    """Checks that a model generates reasonable outputs for two given inputs.

    Args:
        model (torch.nn.Module): The model for which outputs are to be checked. Already loaded to device.
        input1 (torch.Tensor]): Tokenized input1 to check. Already loaded to device.
        input2 (torch.Tensor]): Tokenized input2 to check. Already loaded to device.
        pad_token_id (int): Pad token id for the tokenizer used to generate inputs 1 and 2.

    Returns:
        bool: If the model generates reasonable outputs.
    """
    # Generate 20 tokens of output from the model for each prompt.
    output_length = 20
    # Only take the last 20 tokens since otherwise we also get the prompt ids.
    generate_id1s = model.generate(
        input1,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]
    generate_id2s = model.generate(
        input2,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]

    # Check if too many of the generated ids are the same between the two outputs.
    if torch.sum(torch.eq(generate_id1s, generate_id2s)).item() >= output_length / 2:
        bt.logging.info(
            f"Model with config {model.config} had too much overlap between generated outputs."
        )
        return False

    # Check if internally both responses are too repetitive.
    most_common_counts = []
    for tensor in [generate_id1s, generate_id2s]:
        # Find unique elements and their counts
        _, counts = torch.unique(tensor, return_counts=True)
        # Find the index of the maximum count
        max_count_index = torch.argmax(counts)
        # Extract the count of the most common element
        most_common_counts.append(counts[max_count_index].item())

    if all(count > output_length / 2 for count in most_common_counts):
        bt.logging.info(
            f"Model with config {model.config} had too much repetition in generated outputs."
        )
        return False

    # Passed all the checks, return True.
    return True


def compute_losses(
    model, batches: typing.List[torch.Tensor], device: str, pad_token_id: int
) -> typing.List[float]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (dict): A list of batches.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        pad_token_id int: Pad token id for the tokenizer used to tokenize the batches.

    Returns:
        list: A list of losses for each batch.
    """
    model.to(device)
    model.eval()

    # First check that model generates reasonable looking outputs.
    # Grab 100 tokens from the first two batches as 'prompts'. (1 x Seq Length tensors.)
    prompt_length = 100
    token_inputs_1 = (batches[0][:, :prompt_length]).to(device)
    token_inputs_2 = (batches[1][:, :prompt_length]).to(device)

    if not check_for_reasonable_output(
        model, token_inputs_1, token_inputs_2, pad_token_id
    ):
        return [math.inf for _ in range(len(batches))]

    # Everything looks good! Continue to computing actual losses.

    # Iterate over each page and corresponding batches
    losses = []
    with torch.no_grad():
        for batch in batches:
            try:
                inputs = batch.to(device)
                logits = model(inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                loss = loss_fct(shift_logits, shift_labels).item()

                losses.append(loss)
            except Exception as e:
                bt.logging.error(f"Exception occurred: {e}")
                traceback.print_exc()  # Print the stack trace
                losses.append(math.inf)  # Use infinity to indicate failure

    return losses
