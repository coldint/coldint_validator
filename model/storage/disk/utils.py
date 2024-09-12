import base64
import datetime
import hashlib
import os
import shutil
import sys
import filelock
from model.data import ModelId
from pathlib import Path


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
    dir_hash = hashlib.sha256()

    # Recursively walk everything under the directory for files.
    for cur_path, dirnames, filenames in os.walk(path):
        # Ensure we walk future directories in a consistent order.
        dirnames.sort()
        # Ensure we walk files in a consistent order.
        for filename in sorted(filenames):
            path = os.path.join(cur_path, filename)
            file_hash = get_hash_of_file(path)
            dir_hash.update(file_hash.encode())

    return base64.b64encode(dir_hash.digest()).decode("utf-8")
