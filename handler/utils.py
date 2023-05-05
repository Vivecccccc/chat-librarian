import hashlib
import re

def hash_string(s: str):
    s = re.sub(r"\s+", "", s)
    sha256 = hashlib.sha256()
    sha256.update(s.encode())
    return sha256.hexdigest()