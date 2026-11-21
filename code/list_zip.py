import sys
from remotezip import RemoteZip
url=sys.argv[1]
with RemoteZip(url) as z:
    for i in z.infolist():
        print(f"{i.file_size:>14} {i.compress_size:>14} {i.filename}")
