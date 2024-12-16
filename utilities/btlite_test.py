import os
import sys
import signal
proxy_thread = None
client_handler_threads = []
def stop_threads():
    if proxy_thread:
        client_handler_threads.append(proxy_thread)
    for t in client_handler_threads:
        try:
            if t.is_alive():
                os.kill(t.native_id,signal.SIGTERM)
        except Exception as e:
            print(f'Exception terminating thread: {e}')
    for t in client_handler_threads:
        try:
            t.join()
        except Exception as e:
            print(f'Exception joining thread: {e}')

def shutdown(signum=0,stackframe=None):
    if signum:
        print('SIGINT caught, exiting',file=sys.stderr)
    else:
        print('Shutting down gracefully',file=sys.stderr)
    stop_threads()
    sys.exit(-1)

signal.signal(signal.SIGINT, shutdown)

MESSAGE_TRUNC_LEN = 99
BTLITE = 'btlite'
CHAIN = 'chain'

print('Loading imports...',file=sys.stderr)
import time
import urllib
import select
import socket
import asyncio
import threading
import traceback
import subprocess
import websockets
import numpy as np
import bittensor as bt
import substrateinterface

import btlite

def setup_wallet(uri: str):
    keypair = substrateinterface.Keypair.create_from_uri(uri)
    wallet_path = "/tmp/btlite-wallet-{}".format(uri.strip("/"))
    wallet = bt.wallet(path=wallet_path)
    wallet.set_coldkey(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair, encrypt=False, overwrite=True)
    return wallet

def start_proxy(**kwargs):
    global proxy_thread
    if 'target_websocket' in kwargs:
        proxy_thread = threading.Thread(target=websocket_proxy_listen_wrap, kwargs=kwargs, daemon=False)
    else:
        proxy_thread = threading.Thread(target=tcp_proxy_listen, kwargs=kwargs, daemon=False)
    proxy_thread.start()

def websocket_proxy_listen_wrap(*args,**kwargs):
    asyncio.run(websocket_proxy_listen(*args,**kwargs))

async def websocket_proxy_listen(listen_host='localhost',listen_port=0,**kwargs):
    global websocket_target
    websocket_target = kwargs['target_websocket']
    start_server = await websockets.serve(websocket_proxy_handler, listen_host, listen_port)
    await start_server.wait_closed()
    #asyncio.get_event_loop().run_until_complete(start_server)
    #asyncio.get_event_loop().run_forever()

async def websocket_proxy_handler(client_ws):#,client_path):
    client_path = client_ws.request.path
    try:
        modify = {
                #15:'{"jsonrpc":"2.0","error":{"code":1234,"message":"Mock error","data":"Bummer"},"id":15}',
        }
        idx = 8
        #modify[idx] = '{"jsonrpc":"2.0","error":{"code":1010,"message":"Oh no","data":"Bad things"},"id":%d}'%idx
        stall = []
        stall.append('author_extrinsicUpdate')
        bt.logging.warning(f"Connecting to {websocket_target}")
        async with websockets.connect(websocket_target) as server_ws:
            client_to_server = asyncio.create_task(forward_data(client_ws, server_ws, 'client'))
            server_to_client = asyncio.create_task(forward_data(server_ws, client_ws, 'server', modify=modify,stall=stall))

            done, pending = await asyncio.wait(
                [client_to_server, server_to_client],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
    except Exception as e:
        bt.logging.warning(f"Error in proxy handler: {e}")

async def forward_data(source_ws, target_ws, label='', modify={}, stall=[]):
    cnt = 1
    stalled = False
    try:
        async for message in source_ws:
            bt.logging.info(f'{label} says: #{cnt} {message[:MESSAGE_TRUNC_LEN]}')
            for pat in stall:
                if pat in message and not stalled:
                    bt.logging.warning(f'{label} stalling because of {pat}')
                    stalled = True
            if stalled:
                bt.logging.warning(f'{label} dropping message')
            elif cnt in modify:
                bt.logging.warning(f'{label} replacing message #{cnt} with {modify[cnt]}')
                await target_ws.send(modify[cnt])
            else:
                await target_ws.send(message)
            cnt += 1
    except websockets.exceptions.ConnectionClosed:
        bt.logging.warning('websocket connection closed')
        pass

def tcp_proxy_listen(listen_host='localhost',listen_port=0,**kwargs):
    proxy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy_socket.bind((listen_host, listen_port))
    proxy_socket.listen(5)

    bt.logging.info(f"Proxy listening on {listen_host}:{listen_port}")
    # This only works for plaintext sockets, but is required to connect to the default
    # local testnet subtensor:
    host_orig = b'Host: localhost:'+bytes(str(listen_port),'ASCII')
    host_repl = b'Host: localhost:'+bytes(str(kwargs['target_port']),'ASCII')
    kwargs['replace_map'] = {
        host_orig:host_repl
    }

    while True:
        client_socket, addr = proxy_socket.accept()
        bt.logging.info(f"Accepted connection from {addr}")

        client_handler = threading.Thread(target=proxy_handle_client, args=(client_socket,), kwargs=kwargs)
        client_handler.start()
        client_handler_threads.append(client_handler)

def proxy_handle_client(client_socket,target_host=None,target_port=0,replace_map={},limits={}):
    bt.logging.info(f'connecting {client_socket} to {target_host}:{target_port}...')
    target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        proxy_handle_client_inner(client_socket,target_socket,target_host=target_host,target_port=target_port,replace_map=replace_map,limits=limits)
    except Exception as e:
        bt.logging.info(f'exception handling proxy connection: {e}')
    client_socket.close()
    target_socket.close()

def proxy_handle_client_inner(client_socket,target_socket,target_host=None,target_port=0,replace_map={},limits={}):
    verbose = False
    target_socket.setblocking(1)
    target_socket.connect((target_host, target_port))
    client_socket.setblocking(0)
    target_socket.setblocking(0)

    socknames = {client_socket: BTLITE, target_socket: CHAIN}
    other = {client_socket:target_socket,target_socket:client_socket}
    buffers = {client_socket: b"", target_socket: b""}
    bytes_in = {client_socket: 0, target_socket: 0}
    byte_limit = {client_socket: limits.get(BTLITE,0), target_socket: limits.get(CHAIN,0)}

    hangup = False
    while not hangup or sum([len(b) for b in buffers.values()]):
        wsock = [sock for sock,buf in buffers.items() if len(buf)]
        readable, writable, _ = select.select([client_socket, target_socket], wsock, [])
        if verbose:
            s = [f'{socknames[s]}: {bytes_in[s]} bytes in{" READABLE" if s in readable else ""}{" WRITABLE" if s in writable else ""}' for s in socknames]
            bt.logging.trace(' | '.join(s))

        for sock in readable:
            try:
                data = sock.recv(4096)
                for k,v in replace_map.items():
                    if k not in data:
                        continue
                    data = data.replace(k,v)
                    if verbose:
                        bt.logging.info(f'replaced "{k}" by "{v}" in stream')

                if not data:
                    bt.logging.info(f'{socknames[sock]} socket hung up')
                    return False

                if byte_limit[sock] and bytes_in[sock] + len(data) > byte_limit[sock]:
                    can_send = byte_limit[sock] - bytes_in[sock]
                    if can_send <= 0:
                        return True
                    data = data[:can_send]
                    hangup = True

                bytes_in[sock] += len(data)

#                if total_bytes > BYTE_LIMIT:
#                    print(f"Byte limit exceeded: {BYTE_LIMIT} bytes. Disconnecting or stalling.")
#                    # Stall
#                    select.select([], [], [], STALL_TIME)  # Stall for the specified time
#                    break

                dest = other[sock]
                buffers[dest] += data

            except socket.error as e:
                bt.logging.warning(f'socket error on readable {socknames[sock]} socket: {e}')

        for sock in writable:
            if not buffers[sock]:
                continue
            try:
                sent = sock.send(buffers[sock])
                if verbose:
                    bt.logging.trace(f'sent {sent} bytes of {len(buffers[sock])} byte buffer to {socknames[sock]}')
                else:
                    bt.logging.trace(f'{sent} bytes to {socknames[sock]}, total {bytes_in[other[sock]]}')
                buffers[sock] = buffers[sock][sent:]
            except socket.error as e:
                bt.logging.warning(f'socket error on writable {socknames[sock]} socket: {e}')
    return False

def get_coldkey_pendinghotkeyemission(substrate,coldkey):
    e = 0
    for hotkey in substrate.query('SubtensorModule','OwnedHotkeys',[coldkey]).value:
        e += substrate.query('SubtensorModule','PendingdHotkeyEmission',[hotkey]).value
    return e/1e9

def test_add_args(parser):
    parser.add_argument(
        '--netuid', default=29, type=int
    )
    parser.add_argument(
        '--show-neuron', default=None, type=int,
        help='Print this neuron after connecting subtensor.'
    )
    parser.add_argument(
        '--set-weights', default=None, type=str,
        help='Set comma separated weights on UIDs 0,1,..,n.'
    )
    parser.add_argument(
        '--get-weights', metavar='UID', default=None, type=int,
        help='Get weights set by validator UID.'
    )
    parser.add_argument(
        '--wait-for-inclusion', default=False, action='store_true',
        help='Wait for inclusing when setting weights.'
    )
    parser.add_argument(
        '--wait-for-finalization', default=False, action='store_true',
        help='Wait for inclusing when setting weights.'
    )
    parser.add_argument(
        '--no-await-block', default=False, action='store_true',
        help='When setting weights, for second and later retry, do not wait for block transition.'
    )
    parser.add_argument(
        '--fake-call', default=False, action='store_true',
        help='For testing only! Perform register_network instead of setting weight: this will store an extrinsic in the blockchain for any key, and it will most probably fail anyway.'
    )
    parser.add_argument(
        '--test-alice', default=False, action='store_true',
        help='Shortcut to use local testing subtensor, on Alice port, with Alice wallet. You should start subtensor first using subtensor/scripts/localnet.sh'
    )
    parser.add_argument(
        '--get-pending-emission', metavar='coldkey', default=None, type=str,
        help='Query the runtime for pending emission on hotkeys associated with this coldkey.'
    )
    parser.add_argument(
        '--proxy', default=False, action='store_true',
        help='Provide a proxy between the process and the subtensor, that simply logs packets going back and forth.'
    )
    parser.add_argument(
        '--havoxy', metavar='N', default=0, type=int,
        help='Provide a proxy between the process and the subtensor, that logs packets going back and forth and creates some issues by disrupting the connection after N bytes.'
    )
    parser.add_argument(
        '-w', default=False, action='store_true',
        help='Do not truncate logged messages.'
    )
    parser.add_argument(
        '--sleep', default=False, action='store_true',
        help='Do not stop but sleep indefinitely (for proxy)'
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Tester for btlite.py to test weight setting and pushing other extrinsics to Finney, local subtensor or local test chain.'
    )
    test_add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    # Parse args to get proper logging
    config = bt.config(parser)
    args = parser.parse_args()

    bt.logging(config=config)
    bt.logging.set_debug(True)
    bt.logging.set_trace(True)

    if args.w:
        MESSAGE_TRUNC_LEN = 1024*4

    if args.test_alice:
        # Here arguments are added to sys.argv before parsing.
        # This is a bit crude but easier to write, understand and debug than
        # trying to modify the resulting config structure.

        # Alice and Bob are participating in a local blockchain.
        # Arguments are added for this script, as well as for btcli commands
        # needed to set everything up.

        # Alice will be our validator
        wallet_alice = setup_wallet('//Alice')
        args.netuid = 1
        args_common = {
            '--no_prompt':None,
            '--subtensor.network':'local',
            '--subtensor.chain_endpoint':f'ws://localhost:9945',
            '--netuid':str(args.netuid),
        }

        args_alice = args_common.copy()
        args_alice.update({
            '--wallet.path':wallet_alice.path,
            '--wallet.name':wallet_alice.name,
            '--wallet.hotkey':os.path.basename(wallet_alice.hotkey_file.path),
        })
        args_test = args_alice.copy()
        args_test.update({
            '--set-weights':'0.1,0.3', # let Alice (uid 0) set weights for Alice and Bob (uids 0 and 1)
            '--get-weights':'0', # check weights as set by Alice (uid 0)
        })
        for k,v in args_test.items():
            if not k in sys.argv:
                if v is None:
                    bt.logging.info(f'adding {k}')
                    sys.argv.extend([k])
                else:
                    bt.logging.info(f'adding {k} {v}')
                    sys.argv.extend([k,v])

        # Parse args again with new config
        config = bt.config(parser)
        args = parser.parse_args()

    if args.proxy or args.havoxy:
        proxy_port = 12346
        orig_endpoint = config.subtensor.chain_endpoint
        try:
            parsed = urllib.parse.urlparse(orig_endpoint)
        except Exception as e:
            bt.logging.error(f'Failed to parse chain endpoint: {e}')
            sys.exit(-1)
        new_addr = parsed._replace(netloc=f'localhost:{proxy_port}',scheme='ws')
        config.subtensor.network = config.subtensor.chain_endpoint = urllib.parse.urlunparse(new_addr)
        bt.logging.info(f'Replaced subtensor endpoint {orig_endpoint} with {config.subtensor.chain_endpoint}')
        limits = {}
        if args.havoxy:
            limits[BTLITE] = args.havoxy
            limits[CHAIN] = args.havoxy
        if True:
            # websocket
            start_proxy(
                listen_port=proxy_port,
                listen_host='127.0.0.1',
                target_websocket=orig_endpoint,
            )
        else:
            # TCP
            start_proxy(
                listen_port=proxy_port,
                target_host=parsed.hostname,
                target_port=parsed.port,
                limits=limits,
            )

    bt.logging.info(f'Running bt lite test with config {config}')
    st = btlite.get_subtensor(config=config,no_check=True)

    if False:
        # show there is a bug in substrate
        bt.logging.info(f'st.substrate.metadata={st.substrate.metadata} block_hash={st.substrate.block_hash} {"DEFUNCT" if st.substrate.block_hash and not st.substrate.metadata else ""}')
        try:
            mg = st.metagraph(netuid=1)
        except Exception as e:
            bt.logging.info(f'metagraph sync failed: {e}')
        bt.logging.info(f'st.substrate.metadata={st.substrate.metadata} block_hash={st.substrate.block_hash[:20]}... {"DEFUNCT" if st.substrate.block_hash and not st.substrate.metadata else ""}')
        if st.substrate.metadata is None and st.substrate.block_hash is not None:
            bt.logging.info(f'subtensor is now defunct, attempting to use it for a call')
            try:
                #mg = st.metagraph(netuid=1)
                #bt.logging.info(f'metagraph synced: {type(mg)}')
                #tr = st.substrate.get_type_registry(block_hash=st.substrate.block_hash)
                #bt.logging.info(f'type registry: {tr}')
                m = st.substrate.query_map('SubtensorModule','SubnetOwner')
                bt.logging.info(f'got map: {m}')
            except Exception as e:
                bt.logging.info(f'Exception: {e}\n{traceback.format_exc()}')
            #bt.logging.info(f'metagraph synced? {type(mg)}')
            time.sleep(1)
            shutdown()

    if args.test_alice:
        # Bob will be our miner
        wallet_bob = setup_wallet('//Bob')
        args_bob = args_common.copy()
        args_bob.update({
            '--wallet.path':wallet_bob.path,
            '--wallet.name':wallet_bob.name,
            '--wallet.hotkey':os.path.basename(wallet_bob.hotkey_file.path),
        })
        
        btcli_args_alice = []
        btcli_args_bob = []
        for k,v in args_alice.items():
            btcli_args_alice.extend([k] if v is None else [k,v])
        for k,v in args_bob.items():
            btcli_args_bob.extend([k] if v is None else [k,v])

        # Register a subnet, a validator and a miner
        subnet_available = st.query_module('SubtensorModule','NetworksAdded',None,[args.netuid]).value
        if subnet_available:
            bt.logging.info(f'subnet {args.netuid} already registered, skipping subnet registration')
        else:
            bt.logging.info('registering subnet for Alice...')
            subprocess.run('btcli s create'.split(' ')+btcli_args_alice)
        bt.logging.info('setting subnet weight setting rate limit...')
        subprocess.run('btcli sudo set hyperparameters --param weights_rate_limit --value 0'.split(' ')+btcli_args_alice)
        try:
            mg = st.metagraph(netuid=1)
        except Exception as e:
            bt.logging.info(f'metagraph sync failed: {e}')
        alice_uid = 0
        bob_uid = 1
        if len(mg.hotkeys)<=alice_uid:
            bt.logging.info('registering Alice...')
            subprocess.run('btcli s register'.split(' ')+btcli_args_alice)
            bt.logging.info('staking for Alice...')
            subprocess.run('btcli stake add --amount 10000'.split(' ')+btcli_args_alice)
            bt.logging.info('syncing metagraph...')
            mg.sync(subtensor=st)

        if mg.hotkeys[alice_uid] != wallet_alice.hotkey.ss58_address:
            bt.logging.info(f'Alice is registered, but {mg.hotkeys[alice_uid]} != {wallet_alice.hotkey.ss58_address}?? Aborting.')
            sys.exit(-1)
        else:
            bt.logging.info(f'Alice is registered with {mg.hotkeys[alice_uid]}')

        if len(mg.hotkeys)<=bob_uid:
            bt.logging.info('registering Bob...')
            subprocess.run('btcli s register'.split(' ')+btcli_args_bob)
            bt.logging.info('syncing metagraph...')
            mg.sync(subtensor=st)

        if mg.hotkeys[bob_uid] != wallet_bob.hotkey.ss58_address:
            bt.logging.warning(f'Bob is registered, but {mg.hotkeys[bob_uid]} != {wallet_bob.hotkey.ss58_address}?? Aborting.')
            sys.exit(-1)
        else:
            bt.logging.info(f'Bob is registered with {mg.hotkeys[bob_uid]}')

        sleeptime = 1
        while True:
            vpermits = st.query_module('SubtensorModule','ValidatorPermit',None,[args.netuid]).value
            if vpermits[alice_uid]:
                break
            bt.logging.info('Alice has no validator permit yet, waiting...')
            time.sleep(sleeptime)
            sleeptime *= 1.5


    if args.show_neuron:
        neurons = st.neurons_lite(args.netuid)
        bt.logging.info(f"neurons[{args.show_neuron}]: {neurons[args.show_neuron]}")

    if args.set_weights is not None:
        try:
            wallet = bt.wallet(config=config)
            hotkey = wallet.hotkey
        except Exception as e:
            bt.logging.info(f'failed to open wallet: {type(e).__name__} {e}')
            sys.exit(-1)
        if args.set_weights == "":
            weights = []
        else:
            weights = [float(f) for f in args.set_weights.split(',')]
        uids = list(range(len(weights)))
        bt.logging.info(f'setting weights {weights} on netuid {args.netuid} for uids {uids} with hotkey {str(hotkey)}')
        vali_uid = None
        hotkey_str = hotkey.ss58_address
        for n in st.neurons_lite(args.netuid):
            if n.hotkey == hotkey_str:
                vali_uid = n.uid
                break
        if vali_uid is None:
            bt.logging.warning(f'hotkey {hotkey_str} not found as registered hotkey in netuid {args.netuid}')
        else:
            vpermits = st.query_module('SubtensorModule','ValidatorPermit',None,[args.netuid]).value
            have_vpermit = vpermits[vali_uid]
            bt.logging.warning(f'hotkey {hotkey_str} registered as UID {vali_uid} in netuid {args.netuid}, {"with" if have_vpermit else "WITHOUT"} vPermit')

        call = btlite.set_weights_retry(
            subtensor=st,
            hotkey=hotkey,
            netuid=args.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=args.wait_for_inclusion,
            wait_for_finalization=args.wait_for_finalization,
            await_block=not args.no_await_block,
            fake_call=args.fake_call,
        )
        success, message = asyncio.run(call)
        bt.logging.info(f'RESULT: success={success}, message="{message}"')

    if args.get_weights is not None:
        try:
            metagraph = st.metagraph(netuid=args.netuid,lite=False)
            bt.logging.info(f'weights[{args.get_weights},:] = {metagraph.weights[args.get_weights,:]}')
        except Exception as e:
            bt.logging.error(f'exception syncing metagraph for weights: {type(e).__name__} {e}')

    if args.get_pending_emission is not None:
        coldkey = args.get_pending_emission
        bt.logging.info('fetching pending emission')
        try:
            amount = get_coldkey_pendinghotkeyemission(st.substrate,coldkey)
        except Exception as e:
            bt.logging.info(f'exception fetching pending emission: {e}, {traceback.format_exc()}')
        bt.logging.info(f'pending amount on {coldkey} = {amount}')

    if args.sleep:
        while True:
            time.sleep(1)

    shutdown()
