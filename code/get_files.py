import sys
from remotezip import RemoteZip
url=sys.argv[1]; out=sys.argv[2]
with RemoteZip(url) as z:
    for n in sys.argv[3:]:
        z.extract(n, out)
        print("extracted", n)
