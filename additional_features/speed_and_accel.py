import sys
sys.path.append('..')
import numpy as np
from helpers import *

def _create_speed_and_acceleration(h5_file, n_chunks=10, overwrite=False, verbose=True):
    """
    n_chunks useless, just to be consistent with other create function
    a[t] = (v[t] - v[t-1]) / dt 
    ===> v[t] = sum_{s=0}^{t} a[s] (+ v[-1] = 0)

    """
    freq = 10
    dt = 1 / freq
    
    # Create datasets if required
    if "accel_norm" in h5_file.keys() and not overwrite:
        return None
    shape, dtype = h5_file["x"].shape, h5_file["x"].dtype
    for name in ["accel_norm", "speed_x", "speed_y", "speed_z", "speed_norm"]:
        try:
            h5_file.create_dataset(name, shape=shape, dtype=dtype)
        except:
            pass
    
    # Initiate subject id
    sid = -1
    for ix in range(shape[0]):
        if sid != h5_file["index"][ix]:
            sid = h5_file["index"][ix]
            speed = np.array([[0, 0, 0]])
            if verbose:
                print_bis(f"SUBJECT #{sid}")
        # acceleration
        accel = np.stack([h5_file[feat][ix] for feat in ("x", "y", "z")], axis=-1)
        h5_file["accel_norm"][ix] = np.linalg.norm(accel, ord=2, axis=1)
        # speed
        speed = speed + np.cumsum(accel, axis=0) * dt
        h5_file["speed_x"][ix] = speed[:, 0]
        h5_file["speed_y"][ix] = speed[:, 1]
        h5_file["speed_z"][ix] = speed[:, 2]
        h5_file["speed_norm"][ix] = np.linalg.norm(speed, ord=2, axis=1)
        # speed for next iteration
        speed = speed[[-1], :]
    return None
        
