import numpy as np, glob, sys, os
root=sys.argv[1]
for f in sorted(glob.glob(root+"/*/*/*_0.npy")):
    a=np.load(f,mmap_mode='r'); print(os.path.relpath(f,root), a.shape, a.dtype)
sb=root+"/selective_b/metadata"
for name in ["mult_a","mult_b"]:
    X=np.concatenate([np.load(f"{sb}/{name}_{i}.npy") for i in range(10)])
    u,c=np.unique(X[:,:4],axis=0,return_counts=True)
    print(name,"unique rows (first 4 coeffs):",len(u)); print(u[:10], c[:10])
fk=np.load(root+"/fixed_key/metadata/mult_b_0.npy")
print("fixed_key b unique (4 coeffs):",len(np.unique(fk[:,:4],axis=0)))
print("fixed_key a unique:",len(np.unique(np.load(root+"/fixed_key/metadata/mult_a_0.npy")[:,:4],axis=0)))
