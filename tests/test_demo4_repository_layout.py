"""Relocated Demo 4 launchers, lesson links and Compose must agree."""

import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Demo4RepositoryLayoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("docker"), "Docker Compose CLI is not installed")
    def test_root_compose_resolves_to_the_relocated_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("DEMO_BUCKET=test-weather\nDEMO3_BUCKET=gs://test-weather/demo3\n")

            def configuration(compose_file):
                result = subprocess.run(
                    ["docker", "compose", "--env-file", str(env_file), "-f", compose_file,
                     "config", "--no-env-resolution", "--format", "json"],
                    cwd=ROOT, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            root = configuration("docker-compose.yml")
            relocated = configuration("demos/demo4/docker/docker-compose.yml")
        self.assertEqual(root["name"], relocated["name"])
        self.assertEqual(root["services"], relocated["services"])
        service = root["services"]["ai_weather"]
        self.assertEqual(service["container_name"], "ai_weather")
        self.assertTrue(any(str(port["published"]) == "8503" and port["target"] == 8501
                            for port in service["ports"]))
        self.assertTrue((Path(service["build"]["context"]) / service["build"]["dockerfile"]).is_file())

    def test_both_compatibility_scripts_forward_arguments_from_any_directory(self):
        for script in ("run.sh", "demos/demo4/run.sh"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                canonical = root / "demos/demo4/docker/run.sh"
                canonical.parent.mkdir(parents=True)
                canonical.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
                target = root / script
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / script, target)
                result = subprocess.run(["bash", str(target), "logs", "with spaces"],
                                        cwd=root.parent, capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["logs", "with spaces"])

    def test_demo1_status_uses_the_actual_demo4_container(self):
        path = ROOT / "demos/demo1/dashboard.py"
        assignment = next(node for node in ast.parse(path.read_text()).body
                          if isinstance(node, ast.Assign) and any(
                              isinstance(target, ast.Name) and target.id == "DEMOS" for target in node.targets))
        demos = ast.literal_eval(assignment.value)
        self.assertEqual(next(demo for demo in demos if demo["n"] == 4)["container"], "ai_weather")

    def test_related_documentation_has_no_merge_markers_or_stale_demo4_commands(self):
        for name in ("demos/demo4/README.md", "docs/_episodes/02-obs-dataReadiness.md",
                     "docs/_episodes/04-weather-model.md"):
            text = (ROOT / name).read_text()
            self.assertNotRegex(text, r"(?m)^(?:<<<<<<< |=======$|>>>>>>> )")
        lesson = (ROOT / "docs/_episodes/04-weather-model.md").read_text()
        self.assertNotIn("bash docker/run.sh", lesson)
        self.assertNotIn("8501:localhost:8501", lesson)
        self.assertIn("8503:127.0.0.1:8503", lesson)
