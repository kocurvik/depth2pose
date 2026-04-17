# Binary layout for encode_result / decode_result:
#   float32 x9: R_err, t_err, f1_err, f2_err, f_err, f1, f2, f1_gt, f2_gt
#   int64   x1: runtime
#   float32 x9: R     (row-major 3x3)
#   float32 x9: R_gt  (row-major 3x3)
#   float32 x3: t
#   float32 x3: t_gt
#   When info_keys is None (legacy):
#       uint32     : number of info scalar fields (n_info)
#       repeated n_info times:
#           uint32  key length; <bytes> utf-8 key; float32 value
#   When info_keys is provided:
#       float32 x n_info: scalar values in sorted-key order (keys stored once in HDF5)
#   uint32     : number of inliers
#   <bytes>    : bit-packed inliers (ceil(n/8) bytes, MSB-first)
import io
import os
import struct

import h5py
import numpy as np

from utils.results import get_results_dir
from utils.system_info import save_metadata

_ENCODE_F32 = ('R_err', 't_err', 'f1_err', 'f2_err', 'f_err', 'f1', 'f2', 'f1_gt', 'f2_gt')
_ENCODE_I64 = ('runtime',)
_ENCODE_ARRAYS = (('R', (3, 3)), ('R_gt', (3, 3)), ('t', (3,)), ('t_gt', (3,)))


def encode_result(result, info_keys=None):
    """Encode a single result dict into a 1-D uint8 numpy array.

    If info_keys is provided, scalar info values are written in that order without
    key strings (keys are stored once externally, e.g. in the HDF5 file).
    """
    buf = io.BytesIO()

    for key in _ENCODE_F32:
        buf.write(struct.pack('<f', result[key]))
    for key in _ENCODE_I64:
        buf.write(struct.pack('<q', result[key]))

    for key, _ in _ENCODE_ARRAYS:
        buf.write(np.asarray(result[key], dtype=np.float32).ravel().tobytes())

    info = result['info']
    if info_keys is None:
        scalar_keys = sorted(k for k in info if k != 'inliers')
        buf.write(struct.pack('<I', len(scalar_keys)))
        for k in scalar_keys:
            k_enc = k.encode('utf-8')
            buf.write(struct.pack('<I', len(k_enc)))
            buf.write(k_enc)
            buf.write(struct.pack('<f', float(np.float32(info[k]))))
    else:
        for k in info_keys:
            buf.write(struct.pack('<f', float(np.float32(info[k]))))

    inliers = np.asarray(info['inliers'], dtype=bool)
    buf.write(struct.pack('<I', len(inliers)))
    if len(inliers) > 0:
        buf.write(np.packbits(inliers).tobytes())

    return np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()


def decode_result(data, info_keys=None):
    """Decode a 1-D uint8 numpy array produced by encode_result back into a result dict.

    If info_keys is provided, scalar info values are read in that order without key
    strings (matching the compact encoding written by encode_result with info_keys).
    """
    buf = io.BytesIO(data.tobytes())
    result = {}

    for key in _ENCODE_F32:
        (result[key],) = struct.unpack('<f', buf.read(4))
    for key in _ENCODE_I64:
        (result[key],) = struct.unpack('<q', buf.read(8))

    for key, shape in _ENCODE_ARRAYS:
        n = int(np.prod(shape))
        result[key] = np.frombuffer(buf.read(n * 4), dtype=np.float32).reshape(shape).copy()

    info = {}
    if info_keys is None:
        (n_info,) = struct.unpack('<I', buf.read(4))
        for _ in range(n_info):
            (klen,) = struct.unpack('<I', buf.read(4))
            k = buf.read(klen).decode('utf-8')
            (v,) = struct.unpack('<f', buf.read(4))
            info[k] = v
    else:
        for k in info_keys:
            (v,) = struct.unpack('<f', buf.read(4))
            info[k] = v

    (n_inliers,) = struct.unpack('<I', buf.read(4))
    if n_inliers > 0:
        packed_bytes = buf.read((n_inliers + 7) // 8)
        info['inliers'] = np.unpackbits(np.frombuffer(packed_bytes, dtype=np.uint8))[:n_inliers].astype(bool)
    else:
        info['inliers'] = np.array([], dtype=bool)

    result['info'] = info
    return result


def save_full_results(args, full_results):
    if not full_results:
        return

    h5_path = get_full_results_h5_path(args)

    info_scalar_keys = sorted(k for k in full_results[0]['info'] if k != 'inliers')

    with h5py.File(h5_path, 'w') as f:
        save_metadata(f)

        f.create_dataset('info_scalar_keys', data=np.array(info_scalar_keys, dtype=object))

        # Index fields stored as categorical (vocab + uint32 indices)
        for key in ('image_name_1', 'image_name_2', 'experiment'):
            values = np.array([r[key] for r in full_results])
            vocab, indices = np.unique(values, return_inverse=True)
            f.create_dataset(f'{key}_vocab', data=vocab.astype(object))
            f.create_dataset(key, data=indices.astype(np.uint32))
        f.create_dataset('iterations', data=np.array([r['iterations'] for r in full_results], dtype=np.uint32))

        # Remaining fields encoded per-result without key strings; stored as variable-length uint8 dataset
        encoded = [encode_result(r, info_keys=info_scalar_keys) for r in full_results]
        ds = f.create_dataset('data', shape=(len(encoded),), dtype=h5py.vlen_dtype(np.uint8))
        for i, arr in enumerate(encoded):
            ds[i] = arr


def get_full_results_h5_path(args):
    results_dir = get_results_dir(args, 'full')
    os.makedirs(results_dir, exist_ok=True)
    h5_path = os.path.join(results_dir, f'{args.depth}.h5')
    return h5_path


def load_full_results(args):
    h5_path = get_full_results_h5_path(args)
    print(f"Loading results from {h5_path}")
    full_results = []
    with h5py.File(h5_path, 'r') as f:
        info_scalar_keys = list(f['info_scalar_keys'].asstr()[:]) if 'info_scalar_keys' in f else None
        vocabs = {key: f[f'{key}_vocab'].asstr()[:] for key in ('image_name_1', 'image_name_2', 'experiment')}
        indices = {key: f[key][:] for key in ('image_name_1', 'image_name_2', 'experiment')}
        iterations = f['iterations'][:]
        data = f['data']

        for i in range(len(iterations)):
            res = decode_result(data[i], info_keys=info_scalar_keys)
            for key in ('image_name_1', 'image_name_2', 'experiment'):
                res[key] = vocabs[key][indices[key][i]]
            res['iterations'] = int(iterations[i])
            full_results.append(res)

    return full_results
