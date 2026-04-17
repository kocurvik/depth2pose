# Binary layout for encode_result / decode_result:
#   float32 x9: R_err, t_err, f1_err, f2_err, f_err, f1, f2, f1_gt, f2_gt
#   int64   x1: runtime
#   float32 x9: R     (row-major 3x3)
#   float32 x9: R_gt  (row-major 3x3)
#   float32 x3: t
#   float32 x3: t_gt
#   float32 x2: info floats  (inlier_ratio, model_score)
#   uint32  x3: info ints    (iterations, num_inliers, refinements)
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
_INFO_F32_KEYS = ('inlier_ratio', 'model_score')
_INFO_U32_KEYS = ('iterations', 'num_inliers', 'refinements')


def encode_result(result):
    """Encode a single result dict into a 1-D uint8 numpy array."""
    buf = io.BytesIO()

    with np.errstate(over='ignore'):
        for key in _ENCODE_F32:
            buf.write(struct.pack('<f', float(np.float32(key))))
    for key in _ENCODE_I64:
        buf.write(struct.pack('<q', result[key]))

    for key, _ in _ENCODE_ARRAYS:
        buf.write(np.asarray(result[key], dtype=np.float32).ravel().tobytes())

    info = result['info']
    # sometimes we get maxfloat model scores, these can be ignored
    with np.errstate(over='ignore'):
        for k in _INFO_F32_KEYS:
            buf.write(struct.pack('<f', float(np.float32(info[k]))))
    for k in _INFO_U32_KEYS:
        buf.write(struct.pack('<I', int(info[k])))

    inliers = np.asarray(info['inliers'], dtype=bool)
    buf.write(struct.pack('<I', len(inliers)))
    if len(inliers) > 0:
        buf.write(np.packbits(inliers).tobytes())

    return np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()


def decode_result(data):
    """Decode a 1-D uint8 numpy array produced by encode_result back into a result dict."""
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
    for k in _INFO_F32_KEYS:
        (v,) = struct.unpack('<f', buf.read(4))
        info[k] = v
    for k in _INFO_U32_KEYS:
        (v,) = struct.unpack('<I', buf.read(4))
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

    with h5py.File(h5_path, 'w') as f:
        save_metadata(f)

        # Index fields stored as categorical (vocab + uint32 indices)
        for key in ('image_name_1', 'image_name_2', 'experiment'):
            values = np.array([r[key] for r in full_results])
            vocab, indices = np.unique(values, return_inverse=True)
            f.create_dataset(f'{key}_vocab', data=vocab.astype(object))
            f.create_dataset(key, data=indices.astype(np.uint32))
        f.create_dataset('iterations', data=np.array([r['iterations'] for r in full_results], dtype=np.uint32))

        # Remaining fields encoded per-result; stored as a variable-length uint8 dataset
        encoded = [encode_result(r) for r in full_results]
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
        vocabs = {key: f[f'{key}_vocab'].asstr()[:] for key in ('image_name_1', 'image_name_2', 'experiment')}
        indices = {key: f[key][:] for key in ('image_name_1', 'image_name_2', 'experiment')}
        iterations = f['iterations'][:]
        data = f['data']

        for i in range(len(iterations)):
            res = decode_result(data[i])
            for key in ('image_name_1', 'image_name_2', 'experiment'):
                res[key] = vocabs[key][indices[key][i]]
            res['iterations'] = int(iterations[i])
            full_results.append(res)

    return full_results