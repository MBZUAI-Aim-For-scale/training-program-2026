#!/usr/bin/env python3
"""Check that this Spark can read the program's Google bucket with its service-account key.

  python3 verify_bucket_access.py [gs://bucket]

Standard library only, like spark_setup.py (which runs `check` as a host step): the key's JWT
is signed with the `openssl` command instead of the google-auth package.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

key_path = '/etc/fleet-gcs-key.json'
BUCKET = "gs://aim4scale_training_2026"
SCOPE = "https://www.googleapis.com/auth/devstorage.read_only"


class BucketError(Exception):
    pass


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def google_error(e: urllib.error.HTTPError) -> str:
    """The message in a Google API error response, e.g. "The specified bucket does not exist."."""
    text = e.read().decode()
    try:
        err = json.loads(text)
        return err.get("error_description") or err["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return text.strip()[:200]


def access_token(key: dict) -> str:
    """Trade the service-account key for an OAuth token (Google's JWT bearer flow)."""
    now = int(time.time())
    header = b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = b64url(json.dumps({"iss": key["client_email"], "scope": SCOPE, "aud": key["token_uri"],
                                "iat": now, "exp": now + 600}).encode())
    unsigned = f"{header}.{claims}".encode()
    with tempfile.NamedTemporaryFile("w", suffix=".pem") as pem:  # created mode 0600
        pem.write(key["private_key"])
        pem.flush()
        signed = subprocess.run(["openssl", "dgst", "-sha256", "-sign", pem.name], input=unsigned,
                                capture_output=True, timeout=30)
    if signed.returncode:
        raise BucketError(f"openssl couldn't sign with the key: {signed.stderr.decode().strip()[:200]}")
    body = urllib.parse.urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                   "assertion": f"{unsigned.decode()}.{b64url(signed.stdout)}"}).encode()
    try:
        with urllib.request.urlopen(key["token_uri"], body, timeout=20) as r:
            return json.load(r)["access_token"]
    except urllib.error.HTTPError as e:
        raise BucketError(f"Google rejected the key ({e.code}): {google_error(e)}")


def check(bucket: str = BUCKET, path: str | None = None) -> str:
    """List one object in the bucket. Returns a one-line summary; raises BucketError if it can't."""
    path = path or key_path
    try:
        with open(path) as f:
            key = json.load(f)
    except FileNotFoundError:
        raise BucketError(f"no key at {path}")
    except (OSError, ValueError) as e:
        raise BucketError(f"can't read the key at {path}: {e}")
    token = access_token(key)
    name = bucket.removeprefix("gs://").split("/")[0]
    url = f"https://storage.googleapis.com/storage/v1/b/{name}/o?maxResults=1&fields=items(name)"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            items = json.load(r).get("items", [])
    except urllib.error.HTTPError as e:
        raise BucketError(f"can't list gs://{name} as {key['client_email']} ({e.code}): {google_error(e)}")
    return f"gs://{name} readable as {key['client_email'].split('@')[0]}" + ("" if items else " (bucket is empty)")


if __name__ == "__main__":
    try:
        print(check(*sys.argv[1:2]))
    except (BucketError, urllib.error.URLError) as e:
        sys.exit(f"FAILED: {e}")
