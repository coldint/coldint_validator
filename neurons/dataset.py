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
import torch
import typing
import random
import time
import requests
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from pprint import pprint


class SubsetFineWebEdu2Loader(IterableDataset):

    name: str = "HuggingFaceFW/fineweb-edu-score-2"    
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 2  # Number of retries
    retry_delay: int = 3  # Seconds to wait between retries
    
    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer=None,
        pack=True,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_rows_per_page = 100
        self.tokenizer = tokenizer
        self.pack = pack

        # Get the dataset configs and their row sizes
        self.configs_data = self.fetch_dataset_configs()

        self.pages = []
        self.buffer = []
        if num_pages:
            self.fetch_random_pages(num_pages)

    def fetch_page(self, page_info, pack=None, tokenize=True):
        """
        Fetch a single page; return True when successful.
        If successful, rows have been appended to self.buffer and page appended to self.pages
        Tokenize samples if tokenize==True
        Pack samples (i.e. concatenate) if pack==True
        """
        if pack is None:
            pack = self.pack

        attempt = 0
        config_name, page, split = page_info
        bt.logging.info(f"Fetching page {page_info}")
        while attempt < self.retry_limit:
            # Create the request parameters
            params = dict(dataset=self.name,
                  config=config_name,
                  split=split,
                  offset=page,
                  limit=self.num_rows_per_page
            )
            
            try:
                response = requests.get(self.rows_base_url, params=params)
                try:
                    response_json = response.json()
                except:
                    response_json = "Invalid JSON"
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    if tokenize:
                        tokenized = self.tokenizer(content, truncation=True)["input_ids"] + [self.tokenizer.eos_token_id]
                    else:
                        # For fetch_data_to_rows
                        tokenized = content

                    if self.pack:
                        self.buffer.extend(tokenized)
                    else:
                        self.buffer.append(tokenized)
                self.pages.append(page_info)
                return True
            
            except requests.exceptions.RequestException as e:
                if response.status_code == 500:
                    # Internal Server Errors are seen regularly (2024-07-21) for particular pages.
                    # Retry does not help, so don't bother, and don't complain loudly.
                    bt.logging.warning(f"skipping page: {e}")
                    return False
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch {page_info}: {e}, {response_json}, retrying ({attempt}/{self.retry_limit})"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(f"Maximum retry limit reached. Unable to fetch {page_info}")
                    return False

    def get_random_pages(self, num_pages):
        '''
        Pick <num_pages> random pages, return list of (config_name, page, split) tuples
        '''
        pages = []
        for _ in range(num_pages):
            # Choose a random config
            config_name = random.choice(list(self.configs_data.keys()))
            # Choose a random page (row)
            page_info = random.randint(0, self.configs_data[config_name]['num_rows'] - 1 - self.num_rows_per_page)
            split = self.configs_data[config_name]['split']
            pages.append((config_name, page_info, split))
        return pages

    def fetch_data_for_pages(self, pages):
        '''
        Clear self.buffer/self.pages and fill with pages
        '''
        self.pages = []
        self.buffer = []
        for page_info in pages:
            self.fetch_page(page_info)

    def fetch_random_pages(self, num_pages):
        '''
        Clear self.buffer/self.pages and fill with num_pages randomly selected pages
        '''
        self.pages = []
        self.buffer = []
        while len(self.pages) < num_pages:
            page_info = self.get_random_pages(num_pages=1)[0]
            self.fetch_page(page_info)

    def fetch_data_to_rows(self, num_pages):
        '''
        Clear self.buffer/self.pages and fill with num_pages randomly selected pages
        '''
        self.pages = []
        self.buffer = []
        while len(self.pages) < num_pages:
            page = self.get_random_pages(num_pages=1)[0]
            self.fetch_page(page, pack=False, tokenize=False)
        return self.buffer

    def tokenize(self, tokenizer, max_len=0):
        """
        Return batches, as tokenized using <tokenizer>
        """
        if type(tokenizer) is str:
            bt.logging.info(f"Loading tokenizer {tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        batches = []
        for irow, row in enumerate(self.buffer):
            ids = tokenizer(row, truncation=False)["input_ids"]
            repro = tokenizer.decode(ids)
            if repro != row:
                raise ValueError(f"Tokenizer did not map back to original for sample {irow}")

            if max_len and len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids += [tokenizer.eos_token_id]
            batches.append(torch.tensor([ids]))

        return batches

    def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different dump names, aka configs, aka samples, of the
        dataset.
        The returned value is a dictionary with dump names as keys and
        a dict of the number of rows and the split as values.
        """
        # Request parameters
        params = dict(dataset=self.name)
        
        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.size_base_url, params=params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Extract the configs dict
                configs_dict = response.json()['size']['splits']

                # Now create a dict with config names (except 'default') as
                # keys, and the number of rows as values
                configs_data = {
                    entry['config']: {
                        'num_rows': entry['num_rows'],
                        'split': entry['split']
                    } for entry in configs_dict if entry['config'] != 'default'
                }

                return configs_data
                    
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch dataset configs: {e}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.buffer) == 0:
            raise StopIteration
        batch = []
        while len(batch) < self.batch_size and len(self.buffer):
            if self.pack:
                batch.append(torch.tensor(self.buffer[: self.sequence_length]))
                self.buffer = self.buffer[self.sequence_length :]
            else:
                # TODO: samples might have to be padded, so now only batch_size == 1 works
                batch.append(torch.tensor(self.buffer.pop(0)))
        return torch.stack(batch)
