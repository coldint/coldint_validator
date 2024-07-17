import asyncio
import unittest
from model import utils
from model.data import Model, ModelId, ModelMetadata
from model.model_tracker import ModelTracker

from model.model_updater import ModelUpdater
from model.storage.disk.disk_model_store import DiskModelStore
from tests.model.storage.fake_model_metadata_store import FakeModelMetadataStore
from tests.model.storage.fake_remote_model_store import FakeRemoteModelStore
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedModel


class TestModelUpdater(unittest.TestCase):
    def setUp(self):
        self.model_tracker = ModelTracker()
        self.local_store = DiskModelStore("test-models")
        self.remote_store = FakeRemoteModelStore()
        self.metadata_store = FakeModelMetadataStore()
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

    def tearDown(self):
        self.local_store.delete_unreferenced_models(dict(), 0)

    def _get_small_model(self) -> PreTrainedModel:
        """Gets a small model that works with even the earliest block."""
        config = GPT2Config(
            n_head=10,
            n_layer=12,
            n_embd=760,
        )
        return GPT2LMHeadModel(config)

    def test_get_metadata(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="test_commit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )

        metadata = asyncio.run(self.model_updater._get_metadata(hotkey))

        self.assertEqual(metadata.id, model_id)
        self.assertIsNotNone(metadata.block)

    def test_sync_model_bad_metadata(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="bad_commit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        # Setup the metadata with a commit that doesn't exist in the remote store.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )

        # Check the model fails to sync but that it doesn't throw an exception.
        self.assertFalse(asyncio.run(self.model_updater.sync_model(hotkey)))

    def test_sync_model_same_metadata(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="test_commit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        pt_model = self._get_small_model()

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata, local, and model_tracker to match.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        self.local_store.store_model(hotkey, model)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)

        asyncio.run(self.model_updater.sync_model(hotkey))

        # Tracker information did not change.
        self.assertEqual(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey),
            model_metadata,
        )

    def test_sync_model_same_metadata_force(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="test_commit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        pt_model = self._get_small_model()

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata, local, and model_tracker to match.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        self.local_store.store_model(hotkey, model)
        # Also setup remote store for redownload.
        asyncio.run(self.remote_store.upload_model(model))

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)

        updated = asyncio.run(self.model_updater.sync_model(hotkey, force=True))

        # Tracker information did not change.
        self.assertEqual(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey),
            model_metadata,
        )

        # We did return updated from the sync_model.
        self.assertTrue(updated)

    def test_sync_model_new_metadata(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="test_commit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        pt_model = self._get_small_model()

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store but not local or the model_tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        asyncio.run(self.remote_store.upload_model(model))

        self.assertIsNone(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        )

        # Our local store raises an exception from the Transformers.from_pretrained method if not found.
        with self.assertRaises(Exception):
            self.local_store.retrieve_model(hotkey, model_id)

        asyncio.run(self.model_updater.sync_model(hotkey))

        self.assertEqual(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey),
            model_metadata,
        )
        self.assertEqual(
            str(self.local_store.retrieve_model(hotkey, model_id)), str(model)
        )

    def test_sync_model_hotkey_hash(self):
        hotkey = "test_hotkey"
        model_hash = "test_hash"
        model_id_chain = ModelId(
            namespace="test_model",
            name="test_name",
            hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="test_commit",
        )
        model_metadata = ModelMetadata(id=model_id_chain, block=1)

        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash=model_hash,
            commit="test_commit",
        )

        pt_model = self._get_small_model()

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store and but not local or the model tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        self.remote_store.inject_mismatched_model(model_id_chain, model)

        # Assert that we do update since the model_updater retries with the hotkey hash as well.
        updated = asyncio.run(self.model_updater.sync_model(hotkey))
        self.assertTrue(updated)

    def test_sync_model_bad_hash(self):
        hotkey = "test_hotkey"
        model_id_chain = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="test_commit",
        )
        model_metadata = ModelMetadata(id=model_id_chain, block=1)

        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="bad_hash",
            commit="test_commit",
        )

        pt_model = self._get_small_model()

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store and but not local or the model tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        self.remote_store.inject_mismatched_model(model_id_chain, model)

        # Assert we do not update due to the hash mismatch between the model in remote store and the metadata on chain.
        updated = asyncio.run(self.model_updater.sync_model(hotkey))
        self.assertFalse(updated)

    def test_sync_model_over_max_parameters(self):
        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="test_commit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        config = GPT2Config(
            n_head=10,
            n_layer=25,  # Increase layer by enough to go over max parameter size.
            n_embd=760,
        )
        pt_model = GPT2LMHeadModel(config)

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store but not local or the model_tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        asyncio.run(self.remote_store.upload_model(model))

        # Assert we do not update due to exceeding the maximum allowed parameter size.
        updated = asyncio.run(self.model_updater.sync_model(hotkey))
        self.assertFalse(updated)

    def test_sync_model_uses_next_model_limit(self):

        # Create a model larger than the limit prior to block 2_405_920.
        pt_model = GPT2LMHeadModel(
            GPT2Config(
                n_head=50,
                n_layer=25,
                n_embd=750,
            )
        )

        hotkey = "test_hotkey"
        model_id = ModelId(
            namespace="test_model",
            name="test_name",
            hash="test_hash",
            commit="test_commit",
        )

        # Upload the large model before the block that uses the new limit.
        model_metadata = ModelMetadata(id=model_id, block=2_405_919)

        model = Model(id=model_id, pt_model=pt_model)

        # Upload the model metadata and model.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        asyncio.run(self.remote_store.upload_model(model))

        # Assert we do not update due to exceeding the maximum allowed parameter size.
        self.assertFalse(asyncio.run(self.model_updater.sync_model(hotkey)))

        # Upload the model again, this time after the block that allows this size of model.
        model_metadata = ModelMetadata(id=model_id, block=2_405_920)
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )

        self.assertTrue(asyncio.run(self.model_updater.sync_model(hotkey)))


if __name__ == "__main__":
    unittest.main()
