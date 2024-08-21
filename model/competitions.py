import json
import requests
from bittensor import logging

required_keys = {"reward", "dataset", "model_types", "model_size", "parameters"}

def validate_competitions(d):
    '''
    Validate competitions dictionary <d>.
    Add defaults from key default.
    Drop comments starting with "_".
    return competitions dictionary if valid, None otherwise.
    '''
    if type(d) is not dict:
        logging.warning("Competitions not a dict")
        return None

    defaults = d.get('default',{})

    ret = {}
    for i, (cname, cinfo) in enumerate(d.items()):
        if cname == 'default':
            continue
        if type(cname) is not str or len(cname)==0 or cname[0]=='_':
            continue

        if type(cinfo) is not dict:
            logging.warning(f"Competition {cname} info not a dict")
            return None

        add_c = defaults.copy()
        add_c.update(cinfo)

        missing_keys = required_keys - set(add_c.keys())
        if len(missing_keys) > 0:
            logging.warning(f"Competition {cname} missing keys {missing_keys}")
            return None

        ret[cname] = add_c

    return ret

def load_competitions(loc):
    '''
    Load competitions from location <loc> (local or URL starting with 'https://'
    Return competitions dictionary, or None on failure
    '''
    try:
        if loc.startswith("https://"):
            req = requests.get(loc)
            req.raise_for_status()
            d = req.json()
        else:
            with open(loc) as f:
                d = json.load(f)
        logging.info(f"Fetched competitions content, containing {len(d)} entries")

    except Exception as e:
        logging.warning(f"Failed to load competitions: {e}")
        return None

    return validate_competitions(d)

def validate_model_constraints(mdl, cparams):
    """
    Validate <mdl> against competition parameters <cparams>
    Return tuple (allowed: bool, reason: str)
    """
    model_type = str(type(mdl))
    valid_type = False
    for allowed_type in cparams['model_types']:
        if allowed_type in model_type:
            valid_type = True
            break

    if not valid_type:
        return False, f"Invalid type {model_type}, allowed: {cparams['model_types']}"

    n_params = model_n_parameters(mdl)
    if n_params > cparams['parameters']:
        return False, f"Too many parameters {n_params} > {cparams['parameters']}"

    return True, "OK"

def model_size(path):
    return sum(os.path.getsize(f"{path}/{f}") for f in os.listdir(path) if os.path.isfile(f"{path}/{f}"))

def model_n_parameters(mdl):
    return sum([p.numel() for p in mdl.parameters()])

def model_get_valid_competitions(mdl, competitions):
    ret = []
    for cname, cinfo in competitions.items():
        if validate_model_constraints(mdl, cinfo)[0]:
            ret.append(cname)
    return ret

if __name__ == "__main__":
    logging.set_debug(True)
    c = load_competitions("../../sn29/competitions.json")

    import os
    import sys
    if len(sys.argv) <= 1:
        sys.exit(0)

    import torch
    import transformers
    logging.info(f"Loading model from {sys.argv[1]}")
    mdl = transformers.AutoModelForCausalLM.from_pretrained(
            sys.argv[1],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            #trust_remote_code=True     # Phi3-small seems to require additional code
    )
    dirsize = model_size(sys.argv[1])
    n_params = model_n_parameters(mdl)
    valid_cs = model_get_valid_competitions(mdl, c)
    logging.info(f"Loaded model: {mdl}")
    logging.info(f"Dirsize: {dirsize}")
    logging.info(f"Parameters: {n_params}")
    logging.info(f"Valid competitions: {valid_cs}")
