import numpy as np

def naninf_filt(ar):
    if ar is None:
        return None
    if not isinstance(ar, np.ndarray):
        ar = np.array(ar)
    mask = ~np.isnan(ar) & ~np.isinf(ar)
    return ar[mask]

def naninf_count(ar):
    if ar is None:
        return 0
    ar = naninf_filt(ar)
    return len(ar)

def naninf_mean(ar):
    if ar is None:
        return 0
    ar = naninf_filt(ar)
    return np.inf if len(ar) == 0 else np.mean(ar)

def naninf_std(ar):
    if ar is None:
        return 0
    ar = naninf_filt(ar)
    return np.inf if len(ar) == 0 else np.std(ar)

def naninf_equal(ar_a,ar_b,close=False):
    if ar_a is None or ar_b is None:
        return False
    if not isinstance(ar_a, np.ndarray):
        ar_a = np.array(ar_a)
    if not isinstance(ar_b, np.ndarray):
        ar_b = np.array(ar_b)
    mask_a = ~np.isnan(ar_a) & ~np.isinf(ar_a)
    mask_b = ~np.isnan(ar_b) & ~np.isinf(ar_b)
    if (mask_a != mask_b).any():
        return False
    if close:
        return np.allclose(ar_a[mask_a],ar_b[mask_a])
    return (ar_a[mask_a] == ar_b[mask_a]).all()

def naninf_close(ar_a,ar_b):
    return naninf_equal(ar_a,ar_b,close=True)

# for informative purposes only
def naninf_meandelta(ar_a,ar_b):
    if ar_a is None or ar_b is None:
        return np.inf
    if not isinstance(ar_a, np.ndarray):
        ar_a = np.array(ar_a)
    if not isinstance(ar_b, np.ndarray):
        ar_b = np.array(ar_b)
    mask_a = ~np.isnan(ar_a) & ~np.isinf(ar_a)
    mask_b = ~np.isnan(ar_b) & ~np.isinf(ar_b)
    mask = mask_a & mask_b
    return np.inf if np.sum(mask) == 0 else np.mean(ar_a[mask]-ar_b[mask])
