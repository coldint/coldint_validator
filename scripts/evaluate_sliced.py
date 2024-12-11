#!/usr/bin/env python

"""
Tool to demonstrate/test the sliced evaluation of models.
"""

import os
import sys
import json
import copy
import time
import logging
import argparse
import subprocess

args = None
tokenizer = None

def imports():
    """
    Delayed imports for known slow imports. This allows argument parsing to happen fast.
    """
    global torch,np,GPT2TokenizerFast,AutoTokenizer,AutoModelForCausalLM
    import numpy as np
    import torch
    import transformers
    from transformers import GPT2TokenizerFast,AutoTokenizer,AutoModelForCausalLM

    import transformers_llama
    import transformers_phi3
    import transformers_phi

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

def evaluate_losses(model,samples):
    """
    Evaluate losses of samples on non-sliced model.
    """
    losses = []
    model.to(args.device)
    for i,sample in enumerate(samples):
        if i >= args.max_samples:
            break
        ids = sample
        if len(ids[0])>args.max_sample_len:
            ids = torch.stack([ids[0][:args.max_sample_len]])
        ids = ids.to(args.device)
        labels = ids.clone()
        logging.debug(f'evaluating sample {i} of length {len(ids[0])}...')
        try:
            out = model(ids, labels=labels, output_hidden_states=False)
            loss = out.loss.detach().item()
            logging.debug(f'loss: {loss}')
            losses.append(loss)
        except Exception as e:
            logging.info(f'failed to evaluate, using inf. Exception: {e}')
            losses.append(np.inf)
    return losses

def load_sample_file(fn):
    """
    Load samples from a file. Support .json, .jsonl or plain text.
    """
    samples = []
    with open(fn,'r') as f:
        if 'jsonl' in fn:
            # enforce one json object per line
            for line in f:
                if len(line) == 0:
                    continue
                js = json.loads(line)
                if type(js) is dict:
                    samples.append(js)
                elif type(js) is str:
                    samples.append({'text':js})
                else:
                    raise Exception("jsonl only supports lines with objects or strings")
        elif 'json' in fn:
            # load a complete json object
            samples = json.load(f)
        else:
            # best effort line based import
            for line in f:
                if len(line) == 0:
                    continue
                try:
                    js = json.loads(line)
                    if type(js) is dict:
                        samples.append(js)
                    else:
                        samples.append({'text':str(js)})
                except:
                    samples.append({'text':line})
    return samples

def load_samples():
    samples = []
    for f in args.samples:
        if f == 'license':
            license_bytes = subprocess.check_output('python -c license.MAXLINES=1<<30;license()'.split(' '))
            license_text = license_bytes.decode('utf-8')
            samples.append({'text':license_text})
            continue
        if not os.path.exists(f):
            logging.error(f'ignoring non-existant sample file {f}')
            continue
        if not os.path.isfile(f):
            logging.error(f'ignoring non-file sample arg {f}')
            continue
        samples.extend(load_sample_file(f))
    if len(args.samples) and not len(samples):
        raise Exception("Failed to load any sample")
    for s in samples:
        if type(s) is not dict:
            raise Exception(f"sample {s} is not a dict")
        if 'ids' in s:
            continue
        if not 'text' in s:
            raise Exception(f"no 'text' key in sample {s}")
        s['ids'] = tokenizer(s['text'])['input_ids']
    # convert to list of tensors of ids
    tensor_samples = []
    for s in samples:
        ids = s['ids']
        ids_tensor = torch.tensor(ids)
        t = torch.stack([ids_tensor])
        tensor_samples.append(t)
    return tensor_samples

def load_model(path, attn_implementation="flash_attention_2"):
    if args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError(f"Unkown datatype {args.dtype}")

    logging.info(f"Loading model {path}, attn={attn_implementation}, dtype {args.dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation=attn_implementation,
        torch_dtype=dtype
    )
    try:
        tokenizer_obj = AutoTokenizer.from_pretrained(args.tokenizer or path)
        logging.info('loaded tokenizer from model path')
    except:
        tokenizer_name = "Xenova/gpt-4"
        logging.info(f'falling back to default tokenizer: {tokenizer_name}')
        tokenizer_obj = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        logging.info('loaded tokenizer')

    return model, tokenizer_obj

def arg_parser(argv):
    parser = argparse.ArgumentParser(description='Slice and evaluate models')
    parser.add_argument('model',
            help='Model/tokenizer directory to process')
    parser.add_argument('samples', default=[], nargs='+',
            help='Sample files in JSON, JSONL or plain text format, pre-tokenized or text')
    parser.add_argument('--max-samples', metavar='N', default=16, type=int,
            help='Evaluate not more than N samples')
    parser.add_argument('--max-sample-len', default=100000, type=int,
            help='Maximum sample length')
    parser.add_argument('--tokenizer', default=None,
            help='Override embedded or default tokenizer')
    parser.add_argument('--attn', default=None, choices=['sdpa','eager','flash_attention_2'],
            help='Override attention implementation when loading model (note that eager and flash_attention_2 are compatible, but the latter requires a cuda device)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16','float16','float32'],
            help='Select model datatype, bfloat16 is default')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
            help='Increase verbosity')
    parser.add_argument('--device', default='cuda:0',
            help='Cuda device to use')
    parser.add_argument('--skip-unsliced-eval', default=False, action='store_true',
            help='Skip unsliced evaluation (e.g. for huge models that will not fit)')
    parser.add_argument('--start-layers', default='0',
            help='List of integers specifying layer starts for each slice (e.g. 0,4,8,12)')
    parser.add_argument('--auto-slice', metavar='N', default=None, type=int,
            help='Automatically slice model in N parts.')

    args = parser.parse_args(argv)

    if args.start_layers is not None:
        args.start_layers = [int(s) for s in args.start_layers.split(',')]

    return args

def main():
    global args, tokenizer
    args = arg_parser(sys.argv[1:])

    logging.logThreads = True
    logging.logProcesses = True
    logging.logMultiprocessing = True
    logformat = logging.Formatter('%(asctime)-15s - %(message)s')
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logformat)
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    logger.addHandler(handler)

    logging.debug('loading imports...')
    imports()
    logging.debug('done loading imports')

    logging.debug(f"Evaluating model {args.model}")

    attn_implementation = args.attn
    if attn_implementation is None:
        if args.device.startswith('cuda'):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "eager"
    if args.skip_unsliced_eval:
        # don't actually load weights yet
        torch.set_default_device('meta')
    t0 = time.time()
    model, tokenizer = load_model(args.model, attn_implementation=attn_implementation)
    t_loading = time.time() - t0
    torch.set_default_device(None)

    if False and args.auto_slice is not None:
        n_layers = model.config.num_hidden_layers
        n_slices = min(args.auto_slice,n_layers)
        args.start_layers = [0]
        while len(args.start_layers)<n_slices:
            last_start = args.start_layers[-1]
            remaining_slices = n_slices-len(args.start_layers)
            remaining_layers = n_layers-last_start
            next_start = last_start + remaining_layers//(remaining_slices+1)
            args.start_layers.append(next_start)

    samples = load_samples()
    logging.debug(f'loaded {len(samples)} samples')

    t_slicing = 0
    t_evaluating_sliced = 0
    with torch.no_grad():
        if args.skip_unsliced_eval:
            losses_regular = 0
            t_evaluating_regular = 0
        else:
            t0 = time.time()
            losses_regular = evaluate_losses(model,samples[:args.max_samples])
            t_evaluating_regular = time.time() - t0
            logging.debug('moving model to cpu')
            model.to('cpu')
            logging.debug(f'evaluated regularly in {t_evaluating_regular}s')
            logging.info(f'losses regular: sum={sum(losses_regular)}, {losses_regular[:20]}...')

        if hasattr(model,'sliced'):
            t0 = time.time()
            sliced = model.sliced(
                    n_slices=args.auto_slice,
                    start_layers=args.start_layers if args.auto_slice is None else None,
                    device=args.device,
            )
            t_slicing = time.time() - t0
            logging.info(f'sliced: {sliced}')

            t0 = time.time()
            losses_sliced = sliced.evaluate_samples(samples[:args.max_samples])
            t_evaluating_sliced = time.time() - t0

            # show loss sum and a few individual losses; should be identical regardless of slicing
            logging.info(f"losses sliced: sum={sum(losses_sliced)}, {losses_sliced[:20]}...")
            logging.info(f"identical: {losses_regular==losses_sliced}")
        else:
            logging.info(f"model doesn't support slicing!")

    logging.info(f'time stats: {t_loading:.01f}s loading, {t_slicing:.01f}s slicing, {t_evaluating_regular:.01f}s/{t_evaluating_sliced:.01f}s evaluating regular/sliced')

if __name__ == '__main__':
    try:
        if not main():
            sys.exit(-1)
        sys.exit(0)
    except KeyboardInterrupt as e:
        pass
