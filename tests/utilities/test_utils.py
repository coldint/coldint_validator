import functools
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
from typing import List, Tuple
import unittest
from unittest import mock
import bittensor as bt

import torch
import constants

from utilities.utils import run_in_subprocess
from utilities import utils


class TestUtils(unittest.TestCase):
    def test_run_in_subprocess(self):
        def test_func(a: int, b: int):
            return a + b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual(3, result)

    def test_run_in_subprocess_timeout(self):
        def test_func(a: int, b: int):
            time.sleep(3)
            return a + b

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(TimeoutError):
            result = run_in_subprocess(func=partial, ttl=1)

    def test_run_in_subprocess_no_return(self):
        def test_func(a: int, b: int):
            pass

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertIsNone(result)

    def test_run_in_subprocess_tuple_return(self):
        def test_func(a: int, b: int):
            return a, b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual((1, 2), result)

    def test_run_in_subprocess_exception(self):
        def test_func(a: int, b: int):
            raise ValueError()

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(ValueError):
            result = run_in_subprocess(func=partial, ttl=5)

    def test_validate_hf_repo_id_too_long(self):
        with self.assertRaises(ValueError) as ve:
            # Max allowed length is 41 characters
            utils.validate_hf_repo_id("my-org/" + "a" * 40)

        self.assertRegex(
            str(ve.exception),
            "Hugging Face repo id must be between 3 and 41 characters",
        )

    def test_validate_hf_repo_id_incorrect_format(self):
        with self.assertRaises(ValueError) as ve:
            utils.validate_hf_repo_id("my-repo-name-without-a-namespace")

        self.assertRegex(
            str(ve.exception), "must be in the format <org or user name>/<repo_name>"
        )

    def test_validate_hf_repo_id_valid(self):
        namespace, name = utils.validate_hf_repo_id("my-org/my-repo-name")
        self.assertEqual("my-org", namespace)
        self.assertEqual("my-repo-name", name)

    def test_save_and_load_version(self):
        version = constants.__spec_version__
        with NamedTemporaryFile() as f:
            self.assertIsNone(utils.get_version(f.name))

            utils.save_version(f.name, version)
            self.assertEqual(utils.get_version(f.name), version)

    def test_move_if_exists_does_not_move_dst_exists(self):
        with NamedTemporaryFile(mode="w+") as f:
            f.write("test")
            f.flush()

            with NamedTemporaryFile() as f2:
                # Destination file exists. Should not move.
                self.assertFalse(utils.move_file_if_exists(f.name, f2.name))
                self.assertEqual(b"", f2.read())
                f.seek(0)
                self.assertEqual(f.read(), "test")

    def test_move_if_exists_does_not_move_src_missing(self):
        with NamedTemporaryFile(mode="w+") as f:
            f.write("test")
            f.flush()

            self.assertFalse(utils.move_file_if_exists("no_file", f.name))

    def test_move_if_exists(self):
        with TemporaryDirectory() as d:
            with open(os.path.join(d, "src"), "w") as f:
                f.write("test")
                f.flush()

                dst = os.path.join(d, "dst")

                self.assertTrue(utils.move_file_if_exists(f.name, dst))
                self.assertFalse(os.path.exists(f.name))
                self.assertTrue(os.path.exists(dst))
                self.assertEqual(open(dst, "rb").read(), b"test")

    def test_get_top_valis(self):
        # Create a metagraph with 10 neurons of varying stake with the top 4 having a validator permit.
        mock_metagraph = mock.MagicMock()
        mock_metagraph.S = torch.tensor(
            [0, 1, 2, 300, 4, 5, 600, 7, 8, 9], dtype=torch.float32
        )
        mock_metagraph.validator_permit = torch.tensor(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
            ],
            dtype=torch.bool,
        )

        # Check top 3.
        self.assertEqual(utils.get_top_valis(mock_metagraph, 3), [6, 3, 9])

        # Check N > valis in the metagraph.
        self.assertEqual(utils.get_top_valis(mock_metagraph, 6), [6, 3, 9, 8])

    def _create_metagraph(self):
        """Returns a mocked metagraph with 2 miners and 2 valis."""
        mock_metagraph = mock.MagicMock()
        stakes = torch.tensor([0, 200, 2, 30], dtype=torch.float32)
        mock_metagraph.S = stakes
        mock_metagraph.validator_permit = stakes >= 30
        return mock_metagraph

    def _neuron_info_with_weights(
        self, uid: int, weights: List[Tuple[int, float]]
    ) -> bt.NeuronInfo:
        return bt.NeuronInfo(
            uid=uid,
            netuid=0,
            active=0,
            stake=bt.Balance.from_rao(0),
            stake_dict={},
            total_stake=bt.Balance.from_rao(0),
            rank=0,
            emission=0,
            incentive=0,
            consensus=0,
            trust=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            weights=weights,
            bonds=[],
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
            pruning_score=0,
        )

    def test_list_top_miners_deduplicated(self):
        """Tests list_top_miners, when validators agree on the top miner."""
        metagraph = self._create_metagraph()

        # Set validator weights such that they agree on miner 0 as the top miner.
        metagraph.neurons = [
            self._neuron_info_with_weights(uid=0, weights=[]),
            self._neuron_info_with_weights(uid=1, weights=[(0, 1)]),
            self._neuron_info_with_weights(uid=2, weights=[]),
            self._neuron_info_with_weights(uid=3, weights=[(0, 1)]),
        ]

        # Verify the miner UID is deduped.
        self.assertSequenceEqual(utils.list_top_miners(metagraph), [0])

    def test_list_top_miners_multiple_miners(self):
        """Tests list_top_miners, when validators disagree on the top miner."""
        metagraph = self._create_metagraph()

        metagraph.neurons = [
            self._neuron_info_with_weights(uid=0, weights=[]),
            self._neuron_info_with_weights(uid=1, weights=[(0, 1)]),
            self._neuron_info_with_weights(uid=2, weights=[]),
            self._neuron_info_with_weights(uid=3, weights=[(2, 1)]),
        ]
        top_miners = utils.list_top_miners(metagraph)
        self.assertEqual(len(top_miners), 2)
        self.assertEqual(set(top_miners), set([0, 2]))

    def test_list_top_miners_multiple_weights_set(self):
        """Tests list_top_miners, when validators assign multiple weights"""
        metagraph = self._create_metagraph()

        # Have vali 1 set multiple weights, ensuring it assigns more than
        # 50% relative weight to UID 0.
        metagraph.neurons = [
            self._neuron_info_with_weights(uid=0, weights=[]),
            self._neuron_info_with_weights(uid=1, weights=[(0, 1), (1, 0.1), (2, 0.5)]),
            self._neuron_info_with_weights(uid=2, weights=[]),
            self._neuron_info_with_weights(uid=3, weights=[]),
        ]
        self.assertEqual(utils.list_top_miners(metagraph), [0])


if __name__ == "__main__":
    unittest.main()
