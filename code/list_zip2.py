import sys
from remotezip import RemoteZip
with RemoteZip(sys.argv[1]) as z:
    for i in z.infolist(): print(i.file_size, i.filename)
