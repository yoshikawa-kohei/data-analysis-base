import hashlib


def strhash(from_str: str) -> str:
    return hashlib.md5(from_str.encode("utf-8")).hexdigest()
