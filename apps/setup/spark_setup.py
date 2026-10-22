#!/usr/bin/env python3
"""Set up a DGX Spark for the ai_weather demos, with a live progress dashboard.

  python3 spark_setup.py                      start the dashboard, check the host, then build,
                                              start and health-check a container for every demo
  python3 spark_setup.py --only demo3         host checks + one demo (repeatable)
  python3 spark_setup.py --rebuild            rebuild demo images even if they exist
  python3 spark_setup.py --host-only          host checks only
  python3 spark_setup.py --no-dashboard       terminal output only
  python3 spark_setup.py status               print the last run's steps
  python3 spark_setup.py dashboard [--stop]   start (or stop) just the dashboard

Standard library only, so it runs on a fresh Spark before anything is installed; the
dashboard's Streamlit goes into .venv/ here on first use. Progress goes to
state/status.json and each step's output to state/logs/<step>.log; dashboard.py reads both.
The demos are listed in demos.toml.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE = HERE / "state"
LOGS = STATE / "logs"
STATUS = STATE / "status.json"
LOCK = STATE / "setup.lock"
VENV = HERE / ".venv"
DASHBOARD_PID = STATE / "dashboard.pid"
GPU_WAIT = Path.home() / ".config/systemd/user/docker.service.d/wait-for-gpu.conf"
MIN_FREE_GB = 150  # one demo image (~50 GB) plus model weights (~90 GB)

# Known failure messages -> what to do. Matched against step output and container errors.
HINTS = [
    (r"unresolvable CDI devices|CDI device injection failed",
     "Docker started before the GPU list existed (common after a reboot): "
     "`systemctl --user restart docker`, then retry."),
    (r"failed to fulfil mount request: open /usr/lib/\S*nvidia\S*: no such file",
     "The GPU list (/var/run/cdi/nvidia.yaml) is out of date after an NVIDIA update: "
     "`sudo systemctl start nvidia-cdi-refresh.service`, then `systemctl --user restart docker`."),
    (r"Cannot connect to the Docker daemon|docker\.sock.*(no such file|connection refused)",
     "Docker isn't running: `systemctl --user start docker`."),
    (r"port is already allocated|address already in use",
     "Another container already uses this port: give each demo its own port in demos.toml."),
    (r"no space left on device",
     "Disk is full: check `docker system df`, then `docker image prune` / `docker builder prune`."),
    (r"pull access denied|unauthorized|authentication required",
     "Registry login needed: `docker login nvcr.io` (user `$oauthtoken`, password = NGC API key)."),
    (r"Temporary failure in name resolution|Could not resolve host|TLS handshake timeout|i/o timeout",
     "Network problem while downloading: check the Spark's internet connection and retry."),
    (r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|OutOfMemoryError",
     "GPU memory exhausted: stop other demo containers (`docker stop <name>`) and retry."),
    (r"is already in use by container",
     "A container with this name exists: `docker rm -f <name>` and retry."),
]

SMOKE_TEST = """
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(1000, 1000, device="cuda")
print("SMOKE matmul ok on", torch.cuda.get_device_name(0), float((x @ x).sum()))
try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
    print("SMOKE flash_attn ok", tuple(flash_attn_func(q, q, q).shape))
except Exception as e:
    print("SMOKE flash_attn unavailable:", e)
"""


class Failed(Exception):
    pass


class Warn(Exception):
    pass


def hint_for(text: str) -> str:
    for pattern, hint in HINTS:
        if re.search(pattern, text, re.I):
            return hint
    return ""


def load_config() -> dict:
    with open(HERE / "demos.toml", "rb") as f:
        return tomllib.load(f)


def tail(path: Path, lines: int = 40) -> str:
    """Last lines of a (possibly huge) log file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64_000))
            return "\n".join(f.read().decode(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


# --- Docker queries (also used by the dashboard) -----------------------------------------

def docker(*args: str, timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def inspect(kind: str, name: str) -> dict | None:
    """`docker <kind> inspect`, or None if it doesn't exist (or Docker is down)."""
    try:
        r = docker(kind, "inspect", name)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return json.loads(r.stdout)[0] if r.returncode == 0 else None


def host_ports(container: dict) -> list[int]:
    """Host ports a container publishes (works for stopped containers too)."""
    bindings = (container.get("HostConfig") or {}).get("PortBindings") or {}
    return sorted({int(b["HostPort"]) for bs in bindings.values() if bs for b in bs if b.get("HostPort")})


def healthy(port: int, timeout: float = 3) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=timeout) as r:
            return r.read().strip() == b"ok"
    except OSError:
        return False


def container_logs(name: str, lines: int = 300) -> str:
    try:
        r = docker("logs", "--tail", str(lines), name)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout + r.stderr


# --- Run bookkeeping ----------------------------------------------------------------------

class Run:
    """The steps of one setup run, saved to status.json after every change."""

    def __init__(self, steps: list[dict], args: list[str]):
        self.data = {
            "pid": os.getpid(), "args": args, "started": time.time(), "finished": None, "state": "running",
            "steps": [dict(s, status="pending", started=None, finished=None, detail="", hint="",
                           progress=None, log=None) for s in steps],
        }
        self._last_save = 0.0
        self.save()

    def update(self, sid: str, throttle: bool = False, **fields):
        next(s for s in self.data["steps"] if s["id"] == sid).update(fields)
        if not throttle or time.time() - self._last_save > 2:
            self.save()

    def status(self, sid: str) -> str:
        return next(s["status"] for s in self.data["steps"] if s["id"] == sid)

    def save(self):
        self._last_save = time.time()
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(STATUS)

    def step(self, sid: str, fn, *args) -> str:
        """Run one step function, logging to logs/<sid>.log. Returns its final status."""
        log_path = LOGS / f"{sid}.log"
        self.update(sid, status="running", started=time.time(), log=log_path.name)
        print(f"  …      {self.title(sid)}", flush=True)
        with open(log_path, "w") as log:
            try:
                status, detail = "ok", fn(self, sid, log, *args) or ""
            except Warn as e:
                status, detail = "warn", str(e)
            except Failed as e:
                status, detail = "failed", str(e)
            except Exception as e:  # a bug or missing tool shouldn't kill the whole run
                status, detail = "failed", f"{type(e).__name__}: {e}"
            log.write(f"\n[{status}] {detail}\n")
        hint = hint_for(detail + "\n" + tail(log_path)) if status != "ok" else ""
        self.update(sid, status=status, detail=detail, hint=hint, finished=time.time(),
                    progress=1.0 if status == "ok" else None)
        self.report(sid)
        return status

    def skip(self, sid: str, why: str):
        self.update(sid, status="skipped", detail=why)
        self.report(sid)

    def title(self, sid: str) -> str:
        return next(s["title"] for s in self.data["steps"] if s["id"] == sid).replace("`", "")

    def report(self, sid: str):
        s = next(s for s in self.data["steps"] if s["id"] == sid)
        label = {"ok": "ok", "warn": "WARN", "failed": "FAILED", "skipped": "skip"}[s["status"]]
        print(f"  {label:6s} {self.title(sid)}: {s['detail']}", flush=True)
        if s["hint"]:
            print(f"         fix: {s['hint']}", flush=True)


def sh(cmd: list[str], log, *, env: dict | None = None, timeout: float | None = None, on_line=None):
    """Run a command from the repo root, streaming its output into the step log."""
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, env={**os.environ, **(env or {})}, text=True, errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    timer = threading.Timer(timeout, proc.kill) if timeout else None
    if timer:
        timer.start()
    recent: list[str] = []
    for line in proc.stdout:
        log.write(line)
        log.flush()
        recent = (recent + [line])[-50:]
        if on_line:
            on_line(line)
    rc = proc.wait()
    if timer:
        timer.cancel()
        if rc == -9:
            raise Failed(f"timed out after {timeout:.0f}s")
    return rc, "".join(recent)


def last_line(out: str) -> str:
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return lines[-1][:300] if lines else "no output"


# --- Host steps (demo1) -------------------------------------------------------------------

def step_gpu(run, sid, log):
    rc, out = sh(["nvidia-smi", "-L"], log, timeout=30)
    if rc:
        raise Failed(f"nvidia-smi failed: {last_line(out)}")
    return out.strip().splitlines()[0]


def step_docker(run, sid, log):
    rc, out = sh(["docker", "info", "--format", "{{.ServerVersion}}|{{.SecurityOptions}}"], log, timeout=30)
    if rc:
        raise Failed(f"Docker not reachable: {last_line(out)}")
    version, opts = out.strip().splitlines()[-1].split("|", 1)
    return f"Docker {version}" + (" (rootless)" if "rootless" in opts else "")


def step_cdi(run, sid, log):
    rc, out = sh(["docker", "run", "--rm", "--device", "nvidia.com/gpu=all", "ubuntu", "nvidia-smi", "-L"],
                 log, timeout=300)
    if rc:
        # Docker ends with "Run 'docker run --help'…"; the useful part is the daemon's error line
        error = next((l for l in out.splitlines() if "Error response from daemon" in l), last_line(out))
        raise Failed(f"containers can't see the GPU: {error.split('error during container init: ')[-1][:300]}")
    return "containers see the GPU"


def step_gpu_wait(run, sid, log):
    """SETUP.md step 1: make rootless Docker wait for the GPU list after a reboot."""
    if GPU_WAIT.exists():
        return "already installed"
    if "rootless" not in docker("info", "--format", "{{.SecurityOptions}}").stdout:
        raise Warn("Docker isn't rootless; skipped (only the Sparks' rootless setup needs this)")
    GPU_WAIT.parent.mkdir(parents=True, exist_ok=True)
    GPU_WAIT.write_text(
        "[Service]\n"
        'ExecStartPre=/bin/sh -c "for i in $(seq 120); do [ -s /var/run/cdi/nvidia.yaml ] && exit 0; '
        'sleep 1; done; exit 0"\n')
    log.write(f"wrote {GPU_WAIT}\n")
    rc, out = sh(["systemctl", "--user", "daemon-reload"], log, timeout=30)
    if rc:
        raise Warn(f"installed {GPU_WAIT.name}, but `systemctl --user daemon-reload` failed "
                   "(run it from a login shell)")
    return "installed: Docker now waits for the GPU after reboots"


def step_disk(run, sid, log):
    root = docker("info", "--format", "{{.DockerRootDir}}").stdout.strip() or str(Path.home())
    free = shutil.disk_usage(root).free / 1e9
    log.write(f"{root}: {free:.0f} GB free\n")
    if free < MIN_FREE_GB:
        raise Warn(f"only {free:.0f} GB free under {root}; demos need ~{MIN_FREE_GB} GB")
    return f"{free:,.0f} GB free"


def step_smoke(run, sid, log, image):
    rc, out = sh(["docker", "run", "--rm", "--device", "nvidia.com/gpu=all", "--ipc=host", image,
                  "python", "-c", SMOKE_TEST], log, timeout=3600)
    if rc or "SMOKE matmul ok" not in out:
        raise Failed(f"PyTorch GPU test failed: {last_line(out)}")
    if "SMOKE flash_attn unavailable" in out:
        raise Warn("CUDA matmul ok; flash-attention not installed in this image")
    return "CUDA matmul + flash-attention ok"


# --- Demo steps ---------------------------------------------------------------------------

BUILD_STEP = re.compile(r"^#\d+ \[(?:[\w.-]+ )?\s*(\d+)/(\d+)\] (.*)")


def step_build(run, sid, log, demo, rebuild):
    image = inspect("image", demo["image"])
    if image and not rebuild:
        log.write(f"image {demo['image']} exists; pass --rebuild to rebuild it\n")
        return f"image present (built {image['Created'][:10]}, {image.get('Size', 0) / 1e9:.0f} GB)"

    def on_line(line):
        m = BUILD_STEP.match(line)
        if m:
            done, total, what = int(m[1]), int(m[2]), m[3]
            run.update(sid, throttle=True, progress=(done - 1) / total,
                       detail=f"layer {done}/{total}: {what[:120]}")

    cmd = ["docker", "build", "--progress=plain", "-t", demo["image"], "-f", demo["dockerfile"], demo["context"]]
    rc, out = sh(cmd, log, on_line=on_line)
    if rc:
        errors = [l for l in out.splitlines() if re.search(r"error|failed", l, re.I)]
        raise Failed(f"docker build failed: {(errors or [last_line(out)])[-1][:300]}")
    return "image built"


def step_start(run, sid, log, demo):
    name = demo["container"]
    c = inspect("container", name)
    if c and c["State"]["Running"]:
        return f"already running (since {c['State']['StartedAt'][:16].replace('T', ' ')})"
    if c:  # stopped or crashed: recreate so it picks up the current image and settings
        log.write(f"container {name} exists but is {c['State']['Status']}; recreating it\n")
        sh(["docker", "rm", "-f", name], log, timeout=60)
    rc, out = sh(demo["run"], log, env=demo.get("env"), timeout=300)
    if rc:
        raise Failed(f"start failed: {last_line(out)}")
    time.sleep(3)
    c = inspect("container", name)
    if not c or not c["State"]["Running"]:
        state = c["State"] if c else {}
        log.write(container_logs(name, 50))
        raise Failed(f"container stopped right after starting: {state.get('Error') or 'exit code ' + str(state.get('ExitCode'))}")
    return "container started"


def step_health(run, sid, log, demo, timeout=240):
    name, port = demo["container"], demo["port"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        c = inspect("container", name)
        if not c or not c["State"]["Running"]:
            log.write(container_logs(name, 80))
            raise Failed(f"container exited (code {c['State']['ExitCode'] if c else '?'}) before the page came up")
        ports = host_ports(c)
        actual = port if port in ports or not ports else ports[0]
        if healthy(actual):
            if actual != port:
                raise Warn(f"page up on :{actual}, but demos.toml says :{port}; recreate the container to move it")
            return f"page up at http://localhost:{port}"
        run.update(sid, throttle=True, detail=f"waiting for http://localhost:{actual} …")
        time.sleep(3)
    log.write(container_logs(name, 80))
    raise Failed(f"page not answering on :{port} after {timeout}s")


# --- Dashboard ----------------------------------------------------------------------------

def dashboard_port(cfg: dict) -> int:
    return int(os.environ.get("DASHBOARD_PORT") or cfg.get("dashboard", {}).get("port", 8500))


def start_dashboard(cfg: dict) -> str:
    """Start dashboard.py in the background unless it's already up; return its URL."""
    port = dashboard_port(cfg)
    url = f"http://localhost:{port}"
    if healthy(port):
        return url
    streamlit = VENV / "bin" / "streamlit"
    if not streamlit.exists():
        print("Installing Streamlit for the dashboard (first run only)…", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        subprocess.run([str(VENV / "bin" / "pip"), "install", "-q", "streamlit==1.61.0"], check=True)
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "dashboard.log", "a") as log:
        proc = subprocess.Popen(
            [str(streamlit), "run", str(HERE / "dashboard.py"), "--server.headless", "true",
             "--server.address", "127.0.0.1", "--server.port", str(port), "--browser.gatherUsageStats", "false"],
            cwd=HERE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)  # outlives this script
    DASHBOARD_PID.write_text(str(proc.pid))
    for _ in range(60):
        if healthy(port):
            return url
        if proc.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError(f"not answering on :{port}; see {STATE / 'dashboard.log'}")


def cmd_dashboard(args, _argv):
    if args.stop:
        pid = DASHBOARD_PID.read_text().strip() if DASHBOARD_PID.exists() else ""
        if pid_alive(pid):
            os.kill(int(pid), signal.SIGTERM)
            print("dashboard stopped")
        DASHBOARD_PID.unlink(missing_ok=True)
        return 0
    print(f"Progress dashboard: {start_dashboard(load_config())}")
    return 0


# --- Main ---------------------------------------------------------------------------------

HOST_STEPS = [
    ("host.gpu", "GPU visible to the host", step_gpu),
    ("host.docker", "Docker daemon running", step_docker),
    ("host.cdi", "Containers can use the GPU", step_cdi),
    ("host.gpu_wait", "Docker waits for GPU after reboot", step_gpu_wait),
    ("host.disk", "Disk space", step_disk),
    ("host.smoke", "PyTorch GPU smoke test", step_smoke),
]


def plan(cfg: dict, only: list[str] | None, host_only: bool) -> tuple[list[dict], list[dict]]:
    demos = [] if host_only else [d for d in cfg["demo"] if not only or d["id"] in only]
    if only and not host_only and len(demos) != len(only):
        sys.exit(f"unknown demo id(s): {set(only) - {d['id'] for d in demos}}")
    steps = [{"id": sid, "group": "demo1", "title": title} for sid, title, _ in HOST_STEPS]
    for d in demos:
        steps += [{"id": f"{d['id']}.build", "group": d["id"], "title": f"Build image `{d['image']}`"},
                  {"id": f"{d['id']}.start", "group": d["id"], "title": f"Start container `{d['container']}`"},
                  {"id": f"{d['id']}.health", "group": d["id"], "title": f"Page answers on :{d['port']}"}]
    return demos, steps


def cmd_run(args, argv):
    cfg = load_config()
    LOGS.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and pid_alive(LOCK.read_text().strip()):
        sys.exit(f"setup already running (pid {LOCK.read_text().strip()})")
    LOCK.write_text(str(os.getpid()))
    try:
        demos, steps = plan(cfg, args.only, args.host_only)
        run = Run(steps, argv)
        if not args.no_dashboard:
            try:
                url = start_dashboard(cfg)
                print(f"Progress dashboard: {url}  (from a laptop: ssh -L {dashboard_port(cfg)}:localhost:"
                      f"{dashboard_port(cfg)} <spark>)\n", flush=True)
            except Exception as e:
                print(f"(dashboard didn't start: {e}; carrying on in the terminal)\n", flush=True)
        print("Spark host (demo1)", flush=True)
        smoke_image = cfg.get("smoke_test", {}).get("image", "nvcr.io/nvidia/pytorch:25.03-py3")
        for sid, title, fn in HOST_STEPS:
            if sid not in ("host.gpu", "host.docker") and run.status("host.docker") == "failed":
                run.skip(sid, "Docker isn't running")
            elif sid == "host.smoke" and run.status("host.cdi") == "failed":
                run.skip(sid, "containers can't see the GPU")
            else:
                run.step(sid, fn, *([smoke_image] if sid == "host.smoke" else []))

        docker_ok = run.status("host.docker") != "failed"
        gpu_ok = run.status("host.cdi") == "ok"
        for d in demos:
            ids = [f"{d['id']}.{s}" for s in ("build", "start", "health")]
            print(f"\n{d['title']}", flush=True)
            if not docker_ok:
                for sid in ids:
                    run.skip(sid, "Docker isn't running")
                continue
            if run.step(ids[0], step_build, d, args.rebuild) == "failed":
                run.skip(ids[1], "image build failed")
                run.skip(ids[2], "image build failed")
            elif not gpu_ok and not (inspect("container", d["container"]) or {}).get("State", {}).get("Running"):
                run.skip(ids[1], "containers can't see the GPU (see host checks)")
                run.skip(ids[2], "container not started")
            elif run.step(ids[1], step_start, d) == "failed":
                run.skip(ids[2], "container not running")
            else:
                run.step(ids[2], step_health, d)

        failed = any(s["status"] == "failed" for s in run.data["steps"])
        run.data.update(state="failed" if failed else "done", finished=time.time())
        run.save()
        print("\nSetup finished with errors (see above, or the dashboard)." if failed else "\nSetup finished.")
        return 1 if failed else 0
    finally:
        LOCK.unlink(missing_ok=True)


def cmd_status(_args, _argv):
    if not STATUS.exists():
        print("no setup run yet")
        return 0
    data = json.loads(STATUS.read_text())
    state = data["state"]
    if state == "running" and not pid_alive(data["pid"]):
        state = "interrupted"
    print(f"last run: {state}, started {time.strftime('%Y-%m-%d %H:%M', time.localtime(data['started']))}")
    for s in data["steps"]:
        print(f"  {s['status']:8s} {s['id']:22s} {s['detail']}")
        if s["hint"]:
            print(f"  {'':8s} {'':22s} fix: {s['hint']}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the setup (the default)")
    r.add_argument("--only", action="append", metavar="DEMO", help="demo id from demos.toml (repeatable)")
    r.add_argument("--rebuild", action="store_true", help="rebuild images even if they exist")
    r.add_argument("--host-only", action="store_true", help="host checks only")
    r.add_argument("--no-dashboard", action="store_true", help="don't start the progress dashboard")
    sub.add_parser("status", help="print the last run's steps")
    dash = sub.add_parser("dashboard", help="start just the progress dashboard")
    dash.add_argument("--stop", action="store_true", help="stop it instead")
    if not argv or argv[0] not in ("run", "status", "dashboard", "-h", "--help"):
        argv = ["run", *argv]
    args = p.parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "dashboard": cmd_dashboard}[args.cmd](args, argv)


if __name__ == "__main__":
    sys.exit(main())
