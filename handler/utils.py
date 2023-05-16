import hashlib
import re

def hash_string(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    sha256 = hashlib.sha256()
    sha256.update(s.encode())
    return sha256.hexdigest()

def hash_int(s: str) -> int:
    s = re.sub(r"\s+", "", s)
    o = hashlib.blake2s(s.encode(), digest_size=6)
    hash_bytes = o.digest()
    return int.from_bytes(hash_bytes, byteorder='big', signed=False)

def hash_components(*args) -> str:
    s = ""
    for arg in args:
        if isinstance(arg, str):
            s += arg
    return hash_string(s)