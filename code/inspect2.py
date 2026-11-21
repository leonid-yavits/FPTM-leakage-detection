import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, h5py
R=f"{ROOT}/dl/ledger/d0nj0n_mlkem_dataset/pqm4-ref/ref/variable_key"
for f in ["leakages/traces_0.npy","metadata/mult_a_0.npy","metadata/mult_b_0.npy"]:
    a=np.load(f"{R}/{f}",mmap_mode='r'); print("ref",f,a.shape,a.dtype, np.asarray(a[:2,:4]).tolist())
M=f"{ROOT}/dl/mkm4/100k_capture_all_2"
for f in ["traces/traces_s0_0.npy","traces/traces_s1_0.npy","metadata/ap_0.npy","metadata/bp_s0_0.npy","metadata/bp_s1_0.npy"]:
    a=np.load(f"{M}/{f}",mmap_mode='r'); print("mkm4",f,a.shape,a.dtype,np.asarray(a[:2,:3]).tolist())
b=(np.load(f"{M}/metadata/bp_s0_0.npy")[:,0].astype(int)+np.load(f"{M}/metadata/bp_s1_0.npy")[:,0].astype(int))%3329
print("mkm4 b0 range",b.min(),b.max(),"unique",len(np.unique(b)))
with h5py.File(f"{ROOT}/dl/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5") as h:
    h.visititems(lambda n,o: print("ascad",n, getattr(o,'shape',''), getattr(o,'dtype','')))
    md=h["Profiling_traces/metadata"]; print(md.dtype); m=md[:5]; print(m)
