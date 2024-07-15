import os
import unittest

from model.data import ModelId, Model
from model.storage.disk.disk_model_store import DiskModelStore
from pretrain.model import get_model

import model.storage.disk.utils as utils


class TestDiskModelStore(unittest.TestCase):
    def setUp(self):
        self.base_dir = "test-models"
        self.disk_store = DiskModelStore(self.base_dir)

    def tearDown(self):
        self.disk_store.delete_unreferenced_models(dict(), 0)

    def test_get_path(self):
        hotkey = "hotkey0"

        expected_path = utils.get_local_miner_dir(self.base_dir, hotkey)
        actual_path = self.disk_store.get_path(hotkey)

        self.assertEqual(expected_path, actual_path)

    def test_store_and_retrieve_model(self):
        hotkey = "hotkey0"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash1",
            commit="TestCommit",
        )

        pt_model = get_model()

        model = Model(id=model_id, pt_model=pt_model)

        # Store the model locally.
        self.disk_store.store_model(hotkey, model)

        # Retrieve the model locally.
        retrieved_model = self.disk_store.retrieve_model(hotkey, model_id)

        # Check that they match.
        self.assertEqual(str(model), str(retrieved_model))

    @unittest.skip(
        "Skip this test by default as it requires flash-attn which requires a compatible gpu."
    )
    def test_store_and_retrieve_optimized_model(self):
        hotkey = "hotkey0"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash1",
            commit="TestCommit",
        )

        pt_model = get_model()

        model = Model(id=model_id, pt_model=pt_model)

        # Store the model locally.
        self.disk_store.store_model(hotkey, model)

        # Retrieve the model locally.
        retrieved_model = self.disk_store.retrieve_model(
            hotkey, model_id, optimized=True
        )

        # Check that they match.
        self.assertEqual(str(model), str(retrieved_model))

    def test_delete_unreferenced_models(self):
        hotkey = "hotkey0"

        # Make 2 model ids with different hashes / commits.
        model_id_1 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash1",
            commit="TestCommit1",
        )
        model_id_2 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash2",
            commit="TestCommit2",
        )

        pt_model = get_model()

        model_1 = Model(id=model_id_1, pt_model=pt_model)
        model_2 = Model(id=model_id_2, pt_model=pt_model)

        # Store both models locally.
        self.disk_store.store_model(hotkey, model_1)
        self.disk_store.store_model(hotkey, model_2)

        # Create the mapping of hotkey to model_id with only the 2nd model.
        valid_models_by_hotkey = dict()
        valid_models_by_hotkey[hotkey] = model_id_2

        # Clear the unreferenced models
        self.disk_store.delete_unreferenced_models(valid_models_by_hotkey, 0)

        # Confirm that model 1 is deleted
        with self.assertRaises(Exception):
            self.disk_store.retrieve_model(hotkey, model_id_1)

        # Confirm that model 2 is still there
        model_2_retrieved = self.disk_store.retrieve_model(hotkey, model_id_2)
        self.assertEqual(str(model_2), str(model_2_retrieved))

    def test_delete_unreferenced_models_removed_hotkey(self):
        hotkey_1 = "hotkey1"
        hotkey_2 = "hotkey2"

        # Make 2 model ids with different hashes / commits.
        model_id_1 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash1",
            commit="TestCommit1",
        )
        model_id_2 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash2",
            commit="TestCommit2",
        )

        pt_model = get_model()

        model_1 = Model(id=model_id_1, pt_model=pt_model)
        model_2 = Model(id=model_id_2, pt_model=pt_model)

        # Store both models locally.
        self.disk_store.store_model(hotkey_1, model_1)
        self.disk_store.store_model(hotkey_2, model_2)

        # Create the mapping of hotkey to model_id with only the 2nd model.
        valid_models_by_hotkey = dict()
        valid_models_by_hotkey[hotkey_2] = model_id_2

        # Clear the unreferenced models
        self.disk_store.delete_unreferenced_models(valid_models_by_hotkey, 0)

        # Confirm that model 1 is deleted
        with self.assertRaises(Exception):
            self.disk_store.retrieve_model(hotkey_1, model_id_1)

        # Confirm that model 2 is still there
        model_2_retrieved = self.disk_store.retrieve_model(hotkey_2, model_id_2)
        self.assertEqual(str(model_2), str(model_2_retrieved))

    def test_delete_unreferenced_models_in_grace(self):
        hotkey = "hotkey0"

        # Make 2 model ids with different hashes / commits.
        model_id_1 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash1",
            commit="TestCommit1",
        )
        model_id_2 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash2",
            commit="TestCommit2",
        )

        pt_model = get_model()

        model_1 = Model(id=model_id_1, pt_model=pt_model)
        model_2 = Model(id=model_id_2, pt_model=pt_model)

        # Store both models locally.
        self.disk_store.store_model(hotkey, model_1)
        self.disk_store.store_model(hotkey, model_2)

        # Create the mapping of hotkey to model_id with only the 2nd model.
        valid_models_by_hotkey = dict()
        valid_models_by_hotkey[hotkey] = model_id_2

        # Clear the unreferenced models
        self.disk_store.delete_unreferenced_models(valid_models_by_hotkey, 60)

        # Confirm that model 1 is still there.
        model_1_retrieved = self.disk_store.retrieve_model(hotkey, model_id_1)
        self.assertEqual(str(model_1), str(model_1_retrieved))

        # Confirm that model 2 is still there
        model_2_retrieved = self.disk_store.retrieve_model(hotkey, model_id_2)
        self.assertEqual(str(model_2), str(model_2_retrieved))

    def test_delete_unreferenced_models_removed_hotkey_in_grace(self):
        hotkey_1 = "hotkey1"
        hotkey_2 = "hotkey2"

        # Make 2 model ids with different hashes / commits.
        model_id_1 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash1",
            commit="TestCommit1",
        )
        model_id_2 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash2",
            commit="TestCommit2",
        )

        pt_model = get_model()

        model_1 = Model(id=model_id_1, pt_model=pt_model)
        model_2 = Model(id=model_id_2, pt_model=pt_model)

        # Store both models locally.
        self.disk_store.store_model(hotkey_1, model_1)
        self.disk_store.store_model(hotkey_2, model_2)

        # Create the mapping of hotkey to model_id with only the 2nd model.
        valid_models_by_hotkey = dict()
        valid_models_by_hotkey[hotkey_2] = model_id_2

        # Clear the unreferenced models
        self.disk_store.delete_unreferenced_models(valid_models_by_hotkey, 60)

        # Confirm that model 1 is still there
        model_1_retrieved = self.disk_store.retrieve_model(hotkey_1, model_id_1)
        self.assertEqual(str(model_1), str(model_1_retrieved))

        # Confirm that model 2 is still there
        model_2_retrieved = self.disk_store.retrieve_model(hotkey_2, model_id_2)
        self.assertEqual(str(model_2), str(model_2_retrieved))

    def test_delete_unreferenced_models_and_unexpected_file(self):
        hotkey = "hotkey0"

        # Make 2 model ids with different hashes / commits.
        model_id_1 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash1",
            commit="TestCommit1",
        )
        model_id_2 = ModelId(
            namespace="TestPath",
            name="TestModel",
            hash="TestHash2",
            commit="TestCommit2",
        )

        pt_model = get_model()

        model_1 = Model(id=model_id_1, pt_model=pt_model)
        model_2 = Model(id=model_id_2, pt_model=pt_model)

        # Store both models locally.
        self.disk_store.store_model(hotkey, model_1)
        self.disk_store.store_model(hotkey, model_2)

        # Also store a random file to the hotkey dir.
        # If the is_dir() check is not correct then we will fail to rmtree this file with '[Errno 20] Not a directory.'
        miners_dir = utils.get_local_miner_dir(self.base_dir, hotkey)
        file_name = miners_dir + os.path.sep + "random.txt"
        file = open(file_name, "w")
        file.write("unexpected file.")
        file.close()

        # Create the mapping of hotkey to model_id with only the 2nd model.
        valid_models_by_hotkey = dict()
        valid_models_by_hotkey[hotkey] = model_id_2

        # Clear the unreferenced models
        self.disk_store.delete_unreferenced_models(valid_models_by_hotkey, 0)

        # Confirm that model 1 is deleted
        with self.assertRaises(Exception):
            self.disk_store.retrieve_model(hotkey, model_id_1)

        # Confirm that model 2 is still there
        model_2_retrieved = self.disk_store.retrieve_model(hotkey, model_id_2)
        self.assertEqual(str(model_2), str(model_2_retrieved))


if __name__ == "__main__":
    unittest.main()
