"""Test launcher arguments with fake tools; no model or network is used."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class LauncherTest(unittest.TestCase):
    @contextmanager
    def synthetic_launcher(self):
        """Exercise shell lifecycle with harmless fake server and runner tools."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            script = root / "scripts/run_local_end_to_end.sh"
            shutil.copyfile(ROOT / "scripts/run_local_end_to_end.sh", script)
            (root / "weights.gguf").write_bytes(b"synthetic checkpoint")
            tools = root / "bin"
            tools.mkdir()
            programs = {
                "curl": """import os, sys
from pathlib import Path
sys.exit(0 if Path(os.environ['TEST_READY']).exists() else 1)
""",
                "llama-server": """import os, signal, sys, time
from pathlib import Path
if sys.argv[1:] == ['--version']:
    print('synthetic-build')
    sys.exit(0)
ready = Path(os.environ['TEST_READY'])
Path(os.environ['TEST_LAUNCHED']).touch()
print('SYNTHETIC_SERVER_STARTUP', flush=True)
ready.touch()
def stop(*_args):
    ready.unlink(missing_ok=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.01)
""",
                "python3": """import os, sys
sys.exit(int(os.environ.get('TEST_RUNNER_EXIT', '0')))
""",
            }
            for name, program in programs.items():
                path = tools / name
                path.write_text("#!" + sys.executable + "\n" + program)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
                       TEST_READY=str(root / "ready"), TEST_LAUNCHED=str(root / "launched"))
            yield root, script, env

    def launch(self, root, script, env, *extra):
        return subprocess.run([
            "bash", str(script), "--gguf", str(root / "weights.gguf"),
            "--model-id", "test-model", "--out-dir", str(root / "out"), *extra,
        ], env=env, capture_output=True, text=True, timeout=15)

    def test_help_and_missing_values(self):
        script = ROOT / "scripts/run_local_end_to_end.sh"
        help_result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--out-dir", help_result.stdout)
        self.assertIn("--overwrite-model-output", help_result.stdout)
        result = subprocess.run(["bash", str(script), "--gguf"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing value", result.stderr)

    def test_hf_download_and_launch_provenance_are_forwarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            script = root / "scripts/run_local_end_to_end.sh"
            shutil.copyfile(ROOT / "scripts/run_local_end_to_end.sh", script)
            tools = root / "bin"
            tools.mkdir()
            log = root / "calls.jsonl"
            ready = root / "ready"
            header = "#!" + sys.executable + "\n"
            programs = {
                "hf": """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ['download', 'test/model', 'weights.gguf']
dest = Path(args[args.index('--local-dir') + 1])
dest.mkdir(parents=True, exist_ok=True)
(dest / 'weights.gguf').write_bytes(b'synthetic checkpoint')
""",
                "curl": """import os, sys
from pathlib import Path
sys.exit(0 if Path(os.environ['TEST_READY']).exists() else 1)
""",
                "llama-server": """import os, signal, sys, time
from pathlib import Path
if sys.argv[1:] == ['--version']:
    print('synthetic-test-build')
    sys.exit(0)
ready = Path(os.environ['TEST_READY'])
ready.touch()
def stop(*args):
    ready.unlink(missing_ok=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.05)
""",
                "python3": """import json, os, sys
with open(os.environ['TEST_LOG'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
""",
            }
            for name, program in programs.items():
                path = tools / name
                path.write_text(header + program)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
                       TEST_READY=str(ready), TEST_LOG=str(log))
            result = subprocess.run([
                "bash", str(script), "--hf-repo", "test/model", "--hf-file", "weights.gguf",
                "--model-id", "test-model", "--out-dir", str(root / "out"),
            ], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            runner = next(args for args in calls if args[:2] == ["-m", "deid.run_model"])
            self.assertIn("--checkpoint-file", runner)
            self.assertIn("--server-version=synthetic-test-build", runner)
            command = runner[runner.index("--server-command") + 1:]
            self.assertEqual(command[:2], ["llama-server", "-m"])
            self.assertEqual(command[-1], "--jinja")
            self.assertEqual(command[2], str(root / "models/model/weights.gguf"))
            self.assertFalse(ready.exists())  # cleanup stopped the fake server

    def test_existing_model_is_refused_before_server_or_log_creation(self):
        with self.synthetic_launcher() as (root, script, env):
            model = root / "out/predictions/test-model"
            model.mkdir(parents=True)
            old_log = root / "out/llama-server.log"
            old_log.write_text("retained prior log")
            result = self.launch(root, script, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("model output already exists", result.stderr)
            self.assertFalse((root / "launched").exists())
            self.assertEqual(old_log.read_text(), "retained prior log")
            self.assertEqual(list((root / "out").glob("llama-server.log.*")), [])

    def test_each_launch_preserves_prior_logs_and_uses_private_unique_log(self):
        with self.synthetic_launcher() as (root, script, env):
            out = root / "out"
            out.mkdir(mode=0o755)
            old_log = out / "llama-server.log"
            old_log.write_text("retained prior log")
            old_log.chmod(0o644)
            for _ in range(2):
                result = self.launch(root, script, env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            fresh = list(out.glob("llama-server.log.*"))
            self.assertEqual(len(fresh), 2)
            self.assertEqual(len({p.name for p in fresh}), 2)
            self.assertEqual(old_log.read_text(), "retained prior log")
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            for log in fresh:
                self.assertEqual(log.stat().st_mode & 0o777, 0o600)
                self.assertIn("SYNTHETIC_SERVER_STARTUP", log.read_text())

    def test_log_symlink_target_is_never_opened_even_with_explicit_overwrite(self):
        with self.synthetic_launcher() as (root, script, env):
            (root / "out/predictions/test-model").mkdir(parents=True)
            target = root / "outside.txt"
            target.write_text("do not replace")
            old_log = root / "out/llama-server.log"
            old_log.symlink_to(target)
            result = self.launch(root, script, env, "--overwrite-model-output")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(target.read_text(), "do not replace")
            self.assertTrue(old_log.is_symlink())
            self.assertEqual(len(list((root / "out").glob("llama-server.log.*"))), 1)

    def test_runner_failure_keeps_prior_and_new_logs(self):
        with self.synthetic_launcher() as (root, script, env):
            out = root / "out"
            out.mkdir()
            old_log = out / "llama-server.log"
            old_log.write_text("retained prior log")
            result = self.launch(root, script, dict(env, TEST_RUNNER_EXIT="1"))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(old_log.read_text(), "retained prior log")
            self.assertEqual(len(list(out.glob("llama-server.log.*"))), 1)
            self.assertFalse((root / "ready").exists())

    def test_unsafe_model_id_or_symlinked_output_is_refused_before_launch(self):
        for variant in ("unsafe-id", "root", "root-slash", "root-dot", "predictions", "model",
                        "root-file", "predictions-file", "model-file"):
            with self.subTest(variant=variant), self.synthetic_launcher() as (root, script, env):
                outside = root / "outside"
                outside.mkdir()
                extra = []
                if variant == "unsafe-id":
                    extra = ["--model-id", "../escape"]
                elif variant in ("root", "root-slash", "root-dot"):
                    (root / "out").symlink_to(outside, target_is_directory=True)
                    if variant == "root-slash":
                        extra = ["--out-dir", str(root / "out") + "/"]
                    elif variant == "root-dot":
                        extra = ["--out-dir", str(root / "out") + "/."]
                elif variant == "root-file":
                    (root / "out").write_text("preserve root file")
                elif variant == "predictions-file":
                    (root / "out").mkdir()
                    (root / "out/predictions").write_text("preserve predictions file")
                elif variant == "model-file":
                    (root / "out/predictions").mkdir(parents=True)
                    (root / "out/predictions/test-model").write_text("preserve model file")
                    extra = ["--overwrite-model-output"]
                elif variant == "predictions":
                    (root / "out").mkdir()
                    (root / "out/predictions").symlink_to(outside, target_is_directory=True)
                else:
                    (root / "out/predictions").mkdir(parents=True)
                    (root / "out/predictions/test-model").symlink_to(outside, target_is_directory=True)
                    extra = ["--overwrite-model-output"]
                result = self.launch(root, script, env, *extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "launched").exists())
                self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
