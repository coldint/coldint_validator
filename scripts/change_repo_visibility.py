#!/usr/bin/env python3

import argparse
from huggingface_hub import update_repo_visibility
import sys
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

parser = argparse.ArgumentParser()
parser.add_argument('repo_id', type=str,
        help='HF repo id (user/repo_name)')
parser.add_argument('visibility', type=str,
        choices=['private', 'public'],
        help="Desired visibility")
args = parser.parse_args()
public = args.visibility == "public"

try:
    update_repo_visibility(args.repo_id, private=not public)
except Exception as e:
    print(f"Failed to update repo visibility {e}")
    print("(You might have to use 'huggingface-cli login'")
    sys.exit(-1)

print(f"Succesfully set {args.repo_id} visibility to {args.visibility}")
sys.exit(0)

