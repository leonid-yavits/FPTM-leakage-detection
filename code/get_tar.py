import sys, shutil
from remotezip import RemoteZip
url,name,out=sys.argv[1:4]
with RemoteZip(url) as z, z.open(name) as src, open(out,"wb") as dst:
    shutil.copyfileobj(src,dst,16<<20)
print("ok",out)
