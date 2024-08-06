#!/usr/bin/env python

# Extrinsic signing server using ZMQ

import sys
import zmq
import orjson
import time
import logging
import argparse
import binascii
import bittensor as bt
from scalecodec import GenericExtrinsic,ScaleBytes

DEFAULT_PORT = 12345

# main_loop() is the core loop of this script:
# 1) receive extrinsic to sign from ZMQ, as JSON
# 2) call handle_request()
# 3) send response as JSON to ZMQ

# handle_request() performs the following steps:
# 1) check if the request is a dict
# 2) check if call module and function are whitelisted
# 3) compose call
# 4) sign extrinsic
# on error, an error response is returned.
# Chain interaction is required to get the current nonce.

# Clients should hotpatch substrate before performing an action that emits extrinsics:
#   import extrinsic_signing_server
#   subtensor.substrate.create_signed_extrinsic = extrinsic_signing_server.create_signed_extrinsic_remote

def main_loop():
    while True:
        try:
            logging.warning(f'awaiting request...')
            req_str = zmq_sock.recv_string()
            if req_str == '':
                logging.warning('ignoring empty request')
                continue
            logging.warning(f'received req: {req_str}')
            req = orjson.loads(req_str)
            logging.warning(f'parsed: {req}')
        except Exception as e:
            logging.warning(f'exception receiving req: {e}')
            time.sleep(1)
            continue

        resp = handle_request(req)
        logging.warning(f'sending response: {resp}')
        zmq_sock.send_string(orjson.dumps(resp).decode('ASCII'))

def handle_request(req):
    module_whitelist = ['SubtensorModule']
    function_whitelist = ['set_weights']

    if type(req) is not dict:
        return {'error':'not an object'}

    call = req.get('call',None)
    if call is None:
        return {'error':'no call'}

    call_module = call.get('call_module','SubtensorModule')
    call_function = call.get('call_function',None)
    call_args = call.get('call_args',None)
    if call_module not in module_whitelist:
        return {'error':'blocked'}
    if call_function not in function_whitelist:
        return {'error':'blocked'}

    logging.warning(f'now signing: {call_module} / {call_function} / {call_args}')
    call = substrate.compose_call(call_module=call_module,call_function=call_function,call_params=call_args)
    logging.warning(f'composed call: {call.value_serialized}')
    era = req.get('era',None)
    extrinsic = substrate.create_signed_extrinsic(
        call=call, keypair=signing_key, era=era
    )
    logging.warning(f'created extrinsic: {extrinsic}')
    resp = {
        'extrinsic':{
            'data':str(extrinsic.data),
            'hash':binascii.hexlify(extrinsic.extrinsic_hash).decode('ASCII')
        }
    }
    logging.warning(f'returning: {resp}')

    return resp


# Inject code to perform signing on a signing server.
def create_signed_extrinsic_remote(call=None,keypair=None,era=None):
    try:
        do_create_signed_extrinsic_remote(call,keypair,era)
    except Exception as e:
        bt.logging.warning(f'Exception trying to fetch a remotely signed extrinsic: {e}')
        return None


def do_create_signed_extrinsic_remote(call=None,keypair=None,era=None):
    bt.logging.warning('CALLED TO SIGN EXTRINSIC')
    bt.logging.warning(f'call: {call}')
    bt.logging.warning(f'keypair: {keypair}')
    bt.logging.warning(f'era: {era}')
    zmq_ctx = zmq.Context()
    zmq_sock = zmq_ctx.socket(zmq.PAIR)
    zmq_sock.setsockopt(zmq.CONNECT_TIMEOUT,3000)
    zmq_sock.setsockopt(zmq.SNDTIMEO,3000)
    zmq_sock.setsockopt(zmq.RCVTIMEO,3000)
    zmq_addr = f"tcp://localhost:{DEFAULT_PORT}"
    zmq_sock.connect(zmq_addr)
    req = {
        'call':call.value_serialized,
        'era':era,
    }
    bt.logging.warning(f'Encoding signing request: {req}')
    req_str = orjson.dumps(req).decode('ASCII')
    bt.logging.warning(f'Sending signing request: {req_str}')
    zmq_sock.send_string(req_str)
    resp = orjson.loads(zmq_sock.recv_string())
    if type(resp) is not dict:
        bt.logging.warning(f'Did not receive dict while trying to fetch a remotely signed extrinsic: {resp}')
        return None
    if 'error' in resp:
        bt.logging.warning(f'Received error while trying to fetch a remotely signed extrinsic: {resp["error"]}')
        return None
    if 'extrinsic' not in resp:
        bt.logging.warning(f'Did not receive extrinsic while trying to fetch a remotely signed extrinsic')
        return None
    bt.logging.warning(f'received response: {resp}')
    extrinsic = resp['extrinsic']
    # There seems to be no proper way to unserialize a GenericExtrinsic, but only .data is used after signing.
    ge = GenericExtrinsic()
    ge.data = ScaleBytes(extrinsic['data'])
    if ge.extrinsic_hash != binascii.unhexlify(bytes(extrinsic['hash'],'ASCII')):
        bt.logging.warning(f'Hash mismatch after receiving remotely signed extrinsic')
        return None
    bt.logging.warning(f'returning extrinsic: {ge}')
    return ge


def init_zmq():
    zmq_ctx = zmq.Context()
    zmq_addr = f'tcp://{args.server_addr}:{args.server_port}'
    zmq_sock = zmq_ctx.socket(zmq.PAIR)
    zmq_sock.bind(zmq_addr)
    logging.warning(f'ZMQ socket created')
    return zmq_sock


def init_wallet():
    wallet = bt.wallet(name=args.wallet,hotkey=args.hotkey)
    if args.hotkey is not None:
        if args.wallet_pass:
            wallet._hotkey = wallet.get_hotkey(args.wallet_pass)
        else:
            wallet.hotkey # triggers prompt
        signing_key = wallet.hotkey
    else:
        if args.wallet_pass:
            wallet._coldkey = wallet.get_coldkey(args.wallet_pass)
        else:
            wallet.coldkey # triggers prompt
        signing_key = wallet.coldkey
    return signing_key


def parse_args():
    parser = argparse.ArgumentParser(description='Extrinsic signing server (access to private keys)')
    parser.add_argument('--network', default="local", type=str,
            help='Subtensor to connect to')
    parser.add_argument('--server-addr', default='localhost', type=str,
            help='Listen addr')
    parser.add_argument('--server-port', default=DEFAULT_PORT, type=int,
            help='Listen port')
    parser.add_argument('--wallet', default='default', type=str,
            help='Wallet to use')
    parser.add_argument('--hotkey', default=None, type=str,
            help='If specified, sign using HOTKEY')
    parser.add_argument('--wallet-pass', default=None, type=str,
            help='Wallet password to use (when specified, otherwise prompt, if needed)')
    parser.add_argument('--test', default=None, action='store_true')
    return parser.parse_args()


def capture_bt_logging():
    def bt_console_print(*args):
        try:
            print(*args)
        except Exception as e:
            print("bt_console_print attempted to print, but we failed to relay this...")

    bt.__console__.print = bt_console_print

    class voidctx:
        def __enter__(self):
            pass
        def __exit__(self,*args):
            pass

    def bt_console_status(*args):
        bt_console_print(*args)
        return voidctx()

    bt.__console__.status = bt_console_status


def main():
    global args, substrate, zmq_sock, signing_key

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.WARNING)
    capture_bt_logging()
    args = parse_args()
    zmq_sock = init_zmq()
    signing_key = init_wallet()
    subtensor = bt.subtensor(network=args.network)
    substrate = subtensor.substrate

    if args.test:
        req_str = b'{"call":{"call_module": "SubtensorModule", "call_function": "set_weights", "call_args": {"dests": [0, 1, 5, 44], "weights": [70, 743, 743, 65535], "netuid": 29, "version_key": 1000}},"era":{"period":5}}'
        req = orjson.loads(req_str)
        resp = handle_request(req)
        logging.warning(f'response: {resp}')
        sys.exit(-1)

    main_loop()

if __name__ == '__main__':
    main()
