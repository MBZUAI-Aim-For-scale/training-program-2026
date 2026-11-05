"""Launcher waits for Streamlit and preserves containers during restart."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class Demo4DockerStartupTests(unittest.TestCase):
    def launch(self, action, fail_ready=False, existing=False):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            docker_dir = workspace / "repo/demos/demo4/docker"
            docker_dir.mkdir(parents=True)
            shutil.copyfile(ROOT / "demos/demo4/docker/run.sh", docker_dir / "run.sh")
            (docker_dir / "docker-compose.yml").write_text("services: {}\n")
            env_file = workspace / "repo/.env"
            env_file.write_text(
                "DEMO_BUCKET=test-weather\nDEMO3_BUCKET=gs://test-weather/demo3\n"
                f"DEMO_CACHE={workspace}/cache\nDEMO_OUTPUTS={workspace}/outputs\n"
                f"ROMP_DEMO_OUT_DIR={workspace}/metrics\n")
            fake_bin = workspace / "bin"
            fake_bin.mkdir()
            fake_docker = fake_bin / "docker"
            fake_docker.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_TEST_LOG"
if [ "$1" = "ps" ] && [ "${DOCKER_TEST_EXISTING:-0}" = "1" ]; then
  printf 'ai_weather\\n'
fi
if [ "${DOCKER_TEST_FAIL_READY:-0}" = "1" ]; then
  case "$*" in *--wait*) exit 1 ;; esac
fi
''')
            fake_docker.chmod(0o755)
            log = workspace / "docker.log"
            result = subprocess.run(
                ["bash", "run.sh", action], cwd=docker_dir,
                env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
                     "DOCKER_TEST_LOG": str(log), "DOCKER_TEST_FAIL_READY": str(int(fail_ready)),
                     "DOCKER_TEST_EXISTING": str(int(existing))},
                capture_output=True, text=True, timeout=10, check=False)
            return result, log.read_text(), str(env_file.resolve())

    def test_start_from_relocated_docker_folder_waits_for_health(self):
        result, commands, env_file = self.launch("start")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--env-file {env_file}", commands)
        self.assertIn("up -d --wait --wait-timeout 120", commands)
        self.assertIn("http://127.0.0.1:8503", result.stdout)
        self.assertIn("8503:127.0.0.1:8503", result.stdout)

    def test_restart_keeps_container_and_waits_for_health(self):
        result, commands, _ = self.launch("restart")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("rm -f", commands)
        self.assertIn(" restart\n", commands)
        self.assertIn("up -d --wait --wait-timeout 120", commands)

    def test_unhealthy_start_does_not_report_success(self):
        result, commands, _ = self.launch("start", fail_ready=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--wait", commands)
        self.assertNotIn("[SUCCESS]", result.stdout)

    def test_start_preserves_an_existing_container(self):
        result, commands, _ = self.launch("start", existing=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("rm -f", commands)
