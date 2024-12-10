import base64
import datetime
import hashlib
import os
import shutil
import sys
import filelock
from model.data import ModelId
from pathlib import Path


def storage_state(base_dir=None,config=None):
    if isinstance(base_dir,str):
        base_dir = Path(base_dir)
    if not base_dir.exists():
        raise Exception(f'path {base_dir} does not exist')
    disk_used = sum(f.stat().st_size for f in base_dir.glob('**/*') if f.is_file())
    statvfs = os.statvfs(base_dir)
    gb_in_use = int(round(disk_used/1e9))
    gb_total = int(round(statvfs.f_frsize*statvfs.f_blocks/1e9))
    gb_free = int(round(statvfs.f_bsize*statvfs.f_bavail/1e9))
    gb_to_delete = 0
    gb_space_left = 0

    size_limit = config.model_store_size_gb
    if size_limit>0:
        usage_str = f"{gb_in_use} GB model store on {gb_total} GB disk, limit is {size_limit} GB"
        if gb_in_use >= size_limit:
            gb_to_delete = gb_in_use - size_limit
        else:
            gb_space_left = size_limit - gb_in_use
    else:
        min_free = -size_limit
        usage_str = f"{gb_in_use} GB model store on {gb_total} GB disk, {gb_free} GB free, minimum required is {min_free} GB free"
        if gb_free <= min_free:
            gb_to_delete = min_free - gb_free
        else:
            gb_space_left = gb_free - min_free

    return {
        'gb_total':gb_total,
        'gb_free':gb_free,
        'gb_space_left':gb_space_left,
        'gb_in_use':gb_in_use,
        'gb_to_delete':gb_to_delete,
        'usage_str':usage_str,
    }

def get_local_miners_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "models")


def get_local_miner_dir(base_dir: str, hotkey: str) -> str:
    return os.path.join(get_local_miners_dir(base_dir), hotkey)


# Hugging face stores models under models--namespace--name/snapshots/commit when downloading.
def get_local_model_dir(base_dir: str, hotkey: str, model_id: ModelId) -> str:
    return os.path.join(
        get_local_miner_dir(base_dir, hotkey),
        "models" + "--" + model_id.namespace + "--" + model_id.name,
    )


def get_local_model_snapshot_dir(base_dir: str, hotkey: str, model_id: ModelId) -> str:
    return os.path.join(
        get_local_model_dir(base_dir, hotkey, model_id),
        "snapshots",
        model_id.commit,
    )


def find_lock_pid(file) -> int:
    # Try to get the pid that holds the lock from /proc/locks.
    # Assumptions on the format:
    # - one line per lock
    # - line contains ...:<inode>
    # - first integer is <pid>
    try:
        inode = os.stat(file).st_ino
        with open('/proc/locks','r') as f:
            for line in f:
                pid = None
                hit = False
                parts = line.strip().split(' ')
                for p in parts:
                    if pid is None and re.match('[0-9]+$',p):
                        pid = int(p)
                    if re.match(f'.*:{inode}$',p):
                        hit = True
                if hit:
                    return pid
    except:
        pass
    return None


def scan_locks(path: str) -> bool:
    locks_dir = Path(path) / '.locks'
    for f in locks_dir.glob('**/*'):
        if not f.is_file():
            continue
        fl = filelock.FileLock(f)
        try:
            fl.acquire(blocking=False)
            fl.release()
        except filelock.Timeout:
            # oddly, acquiring a lock non-blocking gives a timeout exception...
            pid = find_lock_pid(f)
            return True,f,pid
    return False,None,None


def get_hf_download_path(local_path: str, model_id: ModelId) -> str:
    return os.path.join(
        local_path,
        "models" + "--" + model_id.namespace + "--" + model_id.name,
        "snapshots",
        model_id.commit,
    )


def get_newest_datetime_under_path(path: str) -> datetime.datetime:
    newest_filetime = 0

    # Check to see if any file at any level was modified more recently than the current one.
    for cur_path, dirnames, filenames in os.walk(path):
        for filename in filenames:
            try:
                path = os.path.join(cur_path, filename)
                mod_time = os.stat(path).st_mtime
                if mod_time > newest_filetime:
                    newest_filetime = mod_time
            except FileNotFoundError:
                pass

    if newest_filetime == 0:
        return datetime.datetime.min

    return datetime.datetime.fromtimestamp(newest_filetime)


def remove_dir_out_of_grace(path: str, grace_period_seconds: int) -> bool:
    """Removes a dir if the last modified time is out of grace period secs. Returns if it was deleted."""
    last_modified = get_newest_datetime_under_path(path)
    grace = datetime.timedelta(seconds=grace_period_seconds)

    if last_modified < datetime.datetime.now() - grace:
        shutil.rmtree(path=path, ignore_errors=True)
        return True

    return False


def realize_symlinks_in_directory(path: str) -> int:
    """Realizes all symlinks in the given directory, moving the linked file to the location. Returns count removed."""
    realized_symlinks = 0

    for cur_path, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.abspath(os.path.join(cur_path, filename))
            # Get path resolving symlinks if encountered
            real_path = os.path.realpath(path)
            # If different then move
            if path != real_path:
                realized_symlinks += 1
                shutil.move(real_path, path)

    return realized_symlinks


def get_hash_of_file(path: str) -> str:
    blocksize = 64 * 1024
    file_hash = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            data = fp.read(blocksize)
            if not data:
                break
            file_hash.update(data)
    return base64.b64encode(file_hash.digest()).decode("utf-8")


def get_hash_of_directory(path: str) -> str:
    '''
    Get hash of files in directory <path>, not recursing into subdirs.
    If no files have been hashed, return None, otherwise return hash.
    '''

    dir_hash = hashlib.sha256()
    n_hashed = 0

    for cur_path, dirnames, filenames in os.walk(path):
        # Ensure we walk files in a consistent order.
        for filename in sorted(filenames):
            path = os.path.join(cur_path, filename)
            file_hash = get_hash_of_file(path)
            dir_hash.update(file_hash.encode())
            n_hashed += 1

    if n_hashed == 0:
        return None

    return base64.b64encode(dir_hash.digest()).decode("utf-8")
