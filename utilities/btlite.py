import time
import asyncio
import traceback
import numpy as np
import bittensor as bt
import substrateinterface
from bittensor.utils import weight_utils

WEBSOCKET_TIMEOUT = 6

def get_subtensor(*args, retries=3, no_check=False, **kwargs):
    '''
    Try to connect to subtensor <retries> times, return None on failure
    '''
    i_try = 0
    while i_try < retries:
        try:
            st = bt.subtensor(*args, **kwargs)
            bt.logging.info(f"Subtensor successfully connected, current block is {st.block}, initializing...")
            # Call init_runtime to make sure substrate is initialized properly.
            # This may raise exceptions, after which metadata is None and the
            # object is defunct.
            if no_check:
                return st
            st.substrate.init_runtime()
            # By way of post-init set a timeout on the substrate websocket, to
            # prevent infinite waiting if the chain dies.
            st.substrate.websocket.socket.settimeout(WEBSOCKET_TIMEOUT)
            if st.substrate.metadata is None:
                # We should probably not reach this even in light of the described bug.
                bt.logging.warning(f'Substrate init failed silently; substrate.metadata=None. Retrying...')
                continue
            return st
        except Exception as e:
            bt.logging.warning(f"Failed to connect to subtensor {i_try+1}/{retries}: {type(e).__name__} {e}")
            if i_try < retries:
                time.sleep(1)
    bt.logging.error(f"Failed to connect to subtensor {retries} times, giving up")
    return None

def check_reconnect(exception=None,subtensor=None):
    if 'Timeout' not in type(exception).__name__:
        return
    bt.logging.warning('substrate websocket timed out; closing and re-opening')
    try:
        subtensor.substrate.close()
        subtensor.substrate.connect_websocket()
        subtensor.substrate.websocket.socket.settimeout(WEBSOCKET_TIMEOUT)
    except Exception as e:
        bt.logging.error(f'failed to reconnect websocket: {type(e).__name__} {e}')

def get_metagraph(subtensor=None, netuid=0, lite=False, reconnect=True):
    try:
        subtensor.substrate.websocket.socket.settimeout(WEBSOCKET_TIMEOUT)
        return subtensor.metagraph(netuid=netuid,lite=lite)
    except Exception as e:
        if reconnect:
            check_reconnect(exception=e,subtensor=subtensor)
        bt.logging.error(f'failed to get metagraph(netuid={netuid}): {type(e).__name__} {e}')
        return None

def get_metadata(subtensor=None, netuid=0, hotkey=None, reconnect=True):
    try:
        subtensor.substrate.websocket.socket.settimeout(WEBSOCKET_TIMEOUT)
        commit_data = subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=None,
        )
        return commit_data.value
    except Exception as e:
        if reconnect:
            check_reconnect(exception=e,subtensor=subtensor)
        bt.logging.error(f'failed to get metadata(netuid={netuid},hotkey={hotkey}): {type(e).__name__} {e}')
        return None

def get_ar_as_type(ar, dtype):
    '''
    Return <ar> as numpy array of type <dtype> if <ar> is a list, <ar> otherwise.
    '''
    if isinstance(ar, list):
        return np.array(ar, dtype=dtype)
    return ar

def set_weights(st, hotkey, uids, weights, netuid, version_key, wait_for_inclusion=False, wait_for_finalization=False, fake_call=False):
    '''
    Set weights using subtensor <st> for hotkey <hotkey>.
    '''
    if len(weights)==0:
        bt.logging.warning(f'attempted to set {len(weights)} for {len(uids)} uids, is this correct?')
        return True, "Setting zero weights is a NOOP"

    uids = get_ar_as_type(uids, np.int64)
    weights = get_ar_as_type(weights, np.float32)
    weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(uids, weights)

    if not fake_call:
        call = st.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_weights",
            call_params={
                "dests": weight_uids,
                "weights": weight_vals,
                "netuid": netuid,
                "version_key": version_key,
            },
        )
    else:
        call = st.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="register_network",
            call_params={
                "immunity_period": 0,
                "reg_allowed": True
            },
        )

    # Period dictates how long the extrinsic will stay as part of waiting pool
    extrinsic = st.substrate.create_signed_extrinsic(
        call=call,
        keypair=hotkey,
        era={"period": 5},
    )
    bt.logging.trace('calling submit_extrinsic()')
    st.substrate.websocket.socket.settimeout(WEBSOCKET_TIMEOUT)
    try:
        response = st.substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    except Exception as e:
        check_reconnect(exception=e,subtensor=st)
        return False, f'Exception submitting extrinsic: {type(e).__name__} {e}'
    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, "Not waiting for finalization or inclusion."

    bt.logging.trace('calling response.process_events()')
    try:
        response.process_events()
    except Exception as e:
        return False, f'Exception processing events: {type(e).__name__} {e}'

    if response.is_success:
        return True, "Successfully set weights."
    else:
        return False, f"Error: {response.error_message}"

async def set_weights_retry(subtensor=None, hotkey=None, uids=[], weights=[], netuid=0, version_key=0xff, wait_for_inclusion=False, wait_for_finalization=False, retries=3, await_block=True, fake_call=False):
    '''
    Retry weight setting <retries> times.
    '''
    i_try = 0
    timeout = time.time()+retries*20
    while i_try < retries:
        try:
            block = subtensor.block
            if i_try != 0 and await_block:
                # On first attempt, just try, but later aim for start of next block.
                bt.logging.debug(f'Awaiting block {block+1} for attempt {i_try+1}/{retries}')
                while block == subtensor.block and time.time()<timeout:
                    await asyncio.sleep(0.1)
                block = subtensor.block
            bt.logging.warning(f'Setting weights, attempt {i_try+1}/{retries}, block={block}...')
            ret, msg = set_weights(subtensor, hotkey, uids, weights, netuid, version_key, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization, fake_call=fake_call)
            if ret is True:
                return ret, msg
            bt.logging.warning(f'set_weights failed: {msg}')
        except Exception as e:
            if ( isinstance(e,substrateinterface.exceptions.SubstrateRequestException)
                 and isinstance(e.args,tuple)
                 and len(e.args)>0
                 and isinstance(e.args[0],dict)
                ):
                code = e.args[0].get('code',-1)
                message = e.args[0].get('message',-1)
                data = e.args[0].get('data',-1)
                bt.logging.warning(f'RPC exception setting weights ({i_try+1}/{retries}): code={code}, message="{message}", data="{data}"')
                if code == 1010:
                    bt.logging.warning(f'Is the hotkey registered in the subnet, with enough stake?')
                elif code == 1014:
                    bt.logging.warning(f"An extrinsic is already in this block{', waiting for the next block' if await_block else ''}")
            else:
                bt.logging.warning(f"set_weights failed ({i_try+1}/{retries}): {type(e).__name__} {e}, {traceback.format_exc()}")
        i_try += 1
    return False, f"set_weights failed {i_try} times"
