#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

def find_streamlit_process_on_port(port: int):
    """Return PID of a process listening on given port, or None."""
    try:
        if sys.platform.startswith("win"):
            cmd = ["netstat", "-ano"]
            output = subprocess.check_output(cmd, text=True)
            for line in output.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return int(part)
        else:
            cmd = ["lsof", "-i", f":{port}", "-t"]
            output = subprocess.check_output(cmd, text=True).strip()
            if output:
                return int(output.splitlines()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return None

def kill_process(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Killed process {pid}")
    except Exception as e:
        print(f"Could not kill process {pid}: {e}")

def run_streamlit(app_path: Path, port: int, host: str, python_exe: Path):
    existing_pid = find_streamlit_process_on_port(port)
    if existing_pid:
        print(f"Found existing Streamlit process (PID {existing_pid}) on port {port}.")
        response = input("Kill it and restart? [y/N]: ").strip().lower()
        if response == "y":
            kill_process(existing_pid)
            import time
            time.sleep(1)
        else:
            print("Exiting.")
            sys.exit(1)

    cmd = [
        str(python_exe), "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--browser.serverAddress", host,
    ]
    
    # Copy host environment and inject configuration variables
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["VERIFICATION_COUNTRY"] = "Ethiopia"
    
    # Resolves the equivalent of $(pwd)/data/diagnostics dynamically
    # based on where the app.py parent directory is.
    project_root = app_path.resolve().parent.parent
    env["VERIFICATION_DATA_DIR"] = str(project_root / "data" / "diagnostics")

    print(f"\nLaunching Streamlit app at: http://{host}:{port}\n")
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nApp stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nStreamlit exited with error: {e}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the Ethiopia Rainfall Onset Dashboard using uv or current env.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("STREAMLIT_PORT", 8502)),
                        help="Port to run Streamlit on (default: 8502 or STREAMLIT_PORT env).")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host to bind (default: localhost).")
    parser.add_argument("--use-uv", action="store_true",
                        help="Use uv to create a virtual environment and install dependencies (requires uv installed).")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    app_file = here / "ui/app.py"
    if not app_file.exists():
        print(f"ERROR: app.py not found at {app_file}")
        return 1

    if args.use_uv:
        # Use uv for fast virtual environment
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: uv is not installed. Please install it from https://github.com/astral-sh/uv")
            return 1

        venv_dir = here / ".venv"
        if sys.platform.startswith("win"):
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python"

        if not venv_dir.exists():
            print(f"Creating virtual environment with uv at {venv_dir}")
            subprocess.run(["uv", "venv", str(venv_dir)], check=True)

        # Install dependencies: look for requirements.txt in current or parent
        req_candidates = [here / "requirements.txt", here.parent / "requirements.txt"]
        req_file = None
        for f in req_candidates:
            if f.exists():
                req_file = f
                break
        if req_file:
            print(f"Installing dependencies from {req_file} using uv pip...")
            subprocess.run(["uv", "pip", "install", "-r", str(req_file), "--python", str(python_exe)], check=True)
        else:
            print("WARNING: requirements.txt not found. Skipping dependency installation.")
        
        # Ensure streamlit is available
        try:
            subprocess.run([str(python_exe), "-m", "streamlit", "--version"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print("Installing streamlit via uv...")
            subprocess.run(["uv", "pip", "install", "streamlit", "--python", str(python_exe)], check=True)
    else:
        # Use current Python environment
        python_exe = Path(sys.executable)
        try:
            subprocess.run([str(python_exe), "-m", "streamlit", "--version"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print("ERROR: Streamlit is not installed in the current Python environment.")
            print("Please install it with: pip install streamlit")
            return 1

    run_streamlit(app_file, port=args.port, host=args.host, python_exe=python_exe)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())