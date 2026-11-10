"""Explicit retries must discard anonymous clients and old gcloud access tokens."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ai_weather import config


class GCSAuthRetryTests(unittest.TestCase):
    def test_retry_reloads_credentials_and_bypasses_gcsfs_instance_cache(self):
        anonymous, authenticated = object(), object()
        constructor = Mock(side_effect=[anonymous, authenticated])
        with patch.object(config, "_fs_cache", [0, 0.0, None]), \
             patch.object(config, "_token_cache", [1.0, "old-access-token"]), \
             patch.object(config, "_auth_token", side_effect=["anon", "renewed-credentials"]) as auth, \
             patch.dict(sys.modules, {"gcsfs": SimpleNamespace(GCSFileSystem=constructor)}):
            self.assertIs(config.gcs_fs(), anonymous)
            self.assertIs(config.gcs_fs(), anonymous)
            self.assertEqual(auth.call_count, 1)
            config.reset_gcs_connection()
            self.assertEqual(config._token_cache, [0.0, ""])
            self.assertIs(config.gcs_fs(), authenticated)
            self.assertIs(config.gcs_fs(), authenticated)
            self.assertEqual(auth.call_count, 2)
        self.assertEqual(constructor.call_args_list[0].kwargs,
                         {"token": "anon", "skip_instance_cache": True})
        self.assertEqual(constructor.call_args_list[1].kwargs,
                         {"token": "renewed-credentials", "skip_instance_cache": True})

    def test_auth_hint_covers_expired_credentials_and_bucket_permissions(self):
        hint = config._auth_hint(PermissionError("401 Anonymous caller"))
        self.assertIn("expired", hint)
        self.assertIn("gcloud auth login", hint)
        self.assertIn("Retry benchmark inputs", hint)
        self.assertEqual(config._auth_hint(TimeoutError("connection timed out")), "")
