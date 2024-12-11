import constants
import dataset
import json
import numpy as np
import os
import shutil
import random
import bittensor as bt
from model.storage.disk import utils as disk_utils
import time

class EvalState(object):
    '''
    Track samples, models, and losses.
    Samples are kept as plain text
    Models are identified by content hash / index
    Loss matrix is a np.array.
    State is serialized in a json and npz file.
    '''

    # Cache for hashes of model directories
    model_dir_hashes = {}

    # Hash of empty/non-existent dir
    VOID_HASH = "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="

    def __init__(self):
        self.sampleset = {'samples': []}
        self.models = []
        self.losses = np.empty([0,0])
        self.first_seen = {}
        self.last_extend_ts = 0

    def set_params(self, params):
        self.params = params.copy()

    def fix_void_hashes(self):
        to_update = {}
        for dirname, hashval in self.model_dir_hashes.items():
            if hashval == self.VOID_HASH:
                bt.logging.info(f'Model at {dirname} has void hash, fixing...')
                to_update[dirname] = None
        self.model_dir_hashes.update(to_update)
        bt.logging.info(f'Fixed-up {len(to_update)} incorrect directory hashes')

    def load_state(self, tgtdir):
        livedir = f"{tgtdir}/evalstate"
        npz_fn = f"{livedir}/eval_losses.npz"
        json_fn = f"{livedir}/eval_state.json"
        if not os.path.exists(npz_fn) or not os.path.exists(json_fn):
            bt.logging.info("No stored EvalState available")
            return None

        loss_npz = np.load(npz_fn)
        if 'losses' not in loss_npz:
            bt.logging.info("EvalState missing 'losses' key")
            return

        with open(json_fn) as f:
            meta = json.load(f)
        if 'sampleset' not in meta or 'models' not in meta or 'model_dir_hashes' not in meta:
            bt.logging.info("EvalState metadata missing keys")
            return

        sh = loss_npz['losses'].shape
        if len(sh) != 2 or sh[0] != len(meta['models']) or sh[1] != len(meta['sampleset']['samples']):
            bt.logging.info(f"EvalState losses incorrect shape ({sh} != {len(meta['models'])} x {len(meta['sampleset']['samples'])}")
            return

        self.losses = loss_npz['losses']
        self.sampleset = meta['sampleset']
        self.models = meta['models']
        self.model_dir_hashes = meta['model_dir_hashes']
        self.last_extend_ts = meta.get('last_extend_ts', 0)
        self.first_seen = meta.get('first_seen', {})

        self.fix_void_hashes()

        bt.logging.info(f"EvalState loaded: {len(self.models)} models, {len(self.sampleset['samples'])} samples, {len(self.model_dir_hashes)} model dir hashes")

    def save_state(self, tgtdir):
        livedir = f"{tgtdir}/evalstate"
        tmpdir = f"{livedir}.tmp"
        olddir = f"{livedir}.old"
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir)

        bt.logging.debug(f"Saving EvalState to {tmpdir}, {np.sum(~np.isnan(self.losses))}/{self.losses.size} loss values set")
        np.savez(f"{tmpdir}/eval_losses.npz", losses=self.losses)
        with open(f"{tmpdir}/eval_state.json", "w") as f:
            f.write(json.dumps(dict(
                sampleset=self.sampleset,
                models=self.models,
                model_dir_hashes=self.model_dir_hashes,
                last_extend_ts=self.last_extend_ts,
                first_seen=self.first_seen,
            )))

        bt.logging.debug(f"EvalState: atomic update to {livedir}")
        # Keep state in olddir until new state is put in place.
        if os.path.exists(olddir):
            shutil.rmtree(olddir)
        if os.path.exists(livedir):
            os.rename(livedir, olddir)
        os.rename(tmpdir, livedir)
        if os.path.exists(olddir):
            shutil.rmtree(olddir)

    def update_sampleset(self):
        '''
        Fill self.eval_state['sampleset'] with n_pool_pages pages.
        '''
        ss = self.sampleset
        pagesize = self.params.get('rows_per_page', 25)
        now = time.time()

        n_samples = 0
        if 'samples' not in ss or len(ss['samples']) == 0:
            ss['samples'] = []
            n_samples = self.params.get('db_min_samples', 1000)
        elif now - self.last_extend_ts > self.params.get('db_extend_interval', 12*3600):
            self.last_extend_ts = now
            n_samples = self.params.get('db_extend_samples', 1000)
        else:
            # Nothing to do
            return

        # Update samples, a list of [dict(page=x, samples=y)]
        bt.logging.info(f"Loading {n_samples} new samples using pages of {pagesize} rows")
        try:
            dataloader = dataset.SubsetFineWebEdu2Loader(
                batch_size=1,
                num_pages=0,
                tokenizer=None,
                pack=False
            )
            dataloader.num_rows_per_page = pagesize
        except requests.exceptions.RequestException as e:
            bt.logging.warning(f"Exception instantiating dataloader: {e}.")
            return

        new_samples = []
        while len(new_samples) < n_samples:
            dataloader.buffer = []
            dataloader.fetch_data_to_rows(1)
            for i_s, s in enumerate(dataloader.buffer):
                new_samples.append(dict(page=dataloader.pages[0], ofs=i_s, sample=s))

        # Shuffle new samples to mix different pages
        random.shuffle(new_samples)
        new_samples = new_samples[:n_samples]
        ss['samples'].extend(new_samples)

        # Grow loss matrix if necessary
        self.update_losses_shape()

        # Drop oldest samples if we have enough
        max_samples = self.params.get('db_max_samples', 10000)
        if len(ss['samples']) > max_samples:
            bt.logging.info(f"Dropping oldest {len(ss['samples'])-max_samples} samples")
            ss['samples'] = ss['samples'][-max_samples:]
            self.losses = self.losses[:,-max_samples:].copy()

        self.last_extend_ts = time.time()

    def update_losses_shape(self):
        '''
        Update self.losses to be of proper size, filling new entries with np.nan
        '''
        tgt_shape = (len(self.models), len(self.sampleset['samples']))

        # Grow if necessary
        if self.losses.shape[0] != tgt_shape[0] or self.losses.shape[1] != tgt_shape[1]:
            bt.logging.debug(f"Growing losses from {self.losses.shape} to {tgt_shape}")
            new_losses = np.full(tgt_shape, np.nan)
            new_losses[:self.losses.shape[0],:self.losses.shape[1]] = self.losses
            self.losses = new_losses

        return self.losses

    def update_models(self, new_models):
        '''
        Update self.models to contain <models>, update loss matrix as well.
        '''
        for mdl in new_models:
            if mdl not in self.models:
                self.models.append(mdl)

        self.update_losses_shape()

    def get_model_idx(self, model_dir):
        model_hash = self.model_dir_hashes.get(model_dir, None)
        if model_hash is None:
            try:
                model_hash = disk_utils.get_hash_of_directory(model_dir)
                if model_hash is None:
                    bt.logging.debug(f"Failed to get hash for {model_dir}")
                    return None
            except:
                return None
            if model_hash not in self.model_dir_hashes:
                bt.logging.debug(f"New hash {model_hash} for {model_dir}")
                self.model_dir_hashes[model_dir] = model_hash
        try:
            return self.models.index(model_hash)
        except:
            # model_hash not yet known
            pass
        idx = len(self.models)
        bt.logging.debug(f"Model hash {model_hash} = index {idx}")
        self.models.append(model_hash)
        self.update_losses_shape()
        return idx

    def get_model_first_seen(self, model_idx, hotkey, block):
        '''
        Return dict with "hotkey" and "block" when model_idx was first seen.
        '''
        if model_idx is None:
            return None

        # Update dict if previous entry has higher or equal block number
        if model_idx not in self.first_seen or self.first_seen[model_idx]['block'] >= block:
            self.first_seen[model_idx] = {
                'block': block,
                'hotkey': hotkey,
            }
        return self.first_seen[model_idx]

    def pick_samples(self, models, n):
        '''
        Pick a subset of samples from the sample page pool, adding maximum new scope

        Arguments:
        models: list of model indices
        n:      number of samples to pick

        Returns:
        dataloader object, with sample_idxs property set to matrix indices
        '''
        dataloader = dataset.SubsetFineWebEdu2Loader(
            batch_size=1,
            num_pages=0,
            tokenizer=None,
            pack=False
        )
        samples = self.sampleset['samples']
        models = [v for v in models if v is not None]
        bt.logging.debug(f"Getting losses for models {models}")
        model_losses = self.losses[models,:]

        # model_losses is the matrix of model x ordered samples
        # We now shuffle these samples, while keeping track of their proper matrix index,
        # so that we don't pick the same ones if all samples have been evaluated
        indexes = np.arange(model_losses.shape[1])
        random.shuffle(indexes)
        sample_nan_count = np.sum(np.isnan(model_losses[:,indexes]),axis=0)
        sample_idx = indexes[np.argsort(sample_nan_count)]

        # Pick n_samples from the back of the index list: samples with most NaNs
        n_samples = self.params['eval_samples']
        dataloader.sample_idxs = sample_idx[-n_samples:]
        total_nans = 0
        for idx in dataloader.sample_idxs:
            dataloader.buffer.append(samples[idx]['sample'])
            total_nans += sample_nan_count[idx]
        bt.logging.debug(f"Selected {len(dataloader.sample_idxs)} samples for {len(models)} models, with {total_nans} NaN losses")

        return dataloader

    def update_loss_values(self, dataloader, models, losses):
        '''
        Update loss matrix

        Arguments:
        dataloader: Dataset, including sample indices
        models: list of model indices (or single index)
        losses: loss matrix (or vector)

        Returns:
        < 0 on error, otherwise number of elements updated
        '''

        if type(models) not in (list, tuple):
            models = [models]
            losses = losses[np.newaxis,:]

        if losses.shape[0] != len(models) or losses.shape[1] != len(dataloader.sample_idxs):
            bt.logging.error(f"update_losses(): invalid losses shape {losses.shape}")
            return -1

        n_updated = 0
        for model_idx, model_loss in zip(models, losses):
            model_n_updated = 0
            mask = ~np.isnan(model_loss)
            sample_idxs = dataloader.sample_idxs[mask]
            if not np.allclose(self.losses[model_idx,sample_idxs], model_loss[mask]):
                self.losses[model_idx,sample_idxs] = model_loss[mask]
                model_n_updated = len(model_loss[mask])
                n_updated += model_n_updated
            model_nans = np.sum(np.isnan(self.losses[model_idx]))
            bt.logging.debug(f"Updated {model_n_updated} losses for model idx {model_idx}, {model_nans}/{self.losses.shape[1]} NaNs remain")

        return n_updated

    def reset_loss_values(self, model_idx):
        '''
        Reset losses for a particular model
        '''
        if model_idx >= self.losses.shape[0]:
            return False

        self.losses[model_idx,:] = np.nan
        bt.logging.info(f"Reset cached losses for model idx {model_idx}")
        return True
