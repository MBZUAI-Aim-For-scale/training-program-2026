"""Shared isolated dashboard process execution and cancellation."""

import os
import subprocess


def spawn_process(argv, env, cwd, timeout, on_line=None, cancel_event=None):
    """
    Run `argv` in `cwd` using posix_spawn on Unix to prevent memory fragmentation and fork-safety issues.
    """
    import select
    import signal
    import time

    from ai_weather.benchmark_job import track_worker, untrack_worker

    started = time.time()

    if not hasattr(os, "posix_spawn"):
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        timed_out = False
        try:
            for line in proc.stdout:
                if on_line:
                    on_line(line.rstrip("\n"))
                if cancel_event is not None and cancel_event.is_set():
                    proc.kill()
                    break
                if time.time() - started > timeout:
                    proc.kill()
                    timed_out = True
                    break
        finally:
            proc.stdout.close()
            proc.wait()
        return (
            130
            if cancel_event is not None and cancel_event.is_set()
            else 124 if timed_out else proc.returncode
        ), timed_out

    read_fd, write_fd = os.pipe()
    actions = [
        (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
        (os.POSIX_SPAWN_DUP2, write_fd, 1),
        (os.POSIX_SPAWN_DUP2, write_fd, 2),
        (os.POSIX_SPAWN_CLOSE, read_fd),
        (os.POSIX_SPAWN_CLOSE, write_fd),
    ]
    shell_argv = [
        "/bin/sh",
        "-c",
        'cd "$1" && shift && exec "$@"',
        "sh",
        str(cwd),
        *map(str, argv),
    ]

    try:
        pid = os.posix_spawn("/bin/sh", shell_argv, env, file_actions=actions, setpgroup=0)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(write_fd)
    track_worker(pid)
    status = None
    termination_attempted = False

    def poll_exit():
        nonlocal status
        if status is None:
            exited, child_status = os.waitpid(pid, os.WNOHANG)
            if exited:
                status = child_status

    def terminate_group():
        nonlocal termination_attempted
        termination_attempted = True
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            poll_exit()
            return
        except PermissionError:
            poll_exit()
            if status is None:
                raise
        deadline = time.time() + 0.2
        while status is None and time.time() < deadline:
            poll_exit()
            if status is None:
                time.sleep(0.02)
        try:
            # The group may already be gone after graceful termination.
            os.killpg(pid, 0)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            poll_exit()
            if status is None:
                raise
            if on_line:
                on_line(
                    "[stop-warning] The main benchmark process exited, but the operating system blocked verification of remaining worker cleanup."
                )

    timed_out = False
    buf = b""
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                terminate_group()
                break
            left = timeout - (time.time() - started)
            if left <= 0:
                terminate_group()
                timed_out = True
                break

            ready, _, _ = select.select([read_fd], [], [], min(left, 0.5))
            if not ready:
                continue

            chunk = os.read(read_fd, 65536)
            if not chunk:
                break

            buf += chunk
            *done, buf = buf.split(b"\n")
            for raw in done:
                if on_line:
                    on_line(raw.decode("utf-8", "replace").rstrip("\r"))

        if buf and on_line:
            on_line(buf.decode("utf-8", "replace").rstrip("\r"))
    except BaseException as exc:
        if isinstance(exc, PermissionError) and on_line:
            on_line(
                f"[stop-warning] The operating system denied process management for PID {pid}. Worker shutdown is not confirmed."
            )
        if not termination_attempted:
            terminate_group()
        raise
    finally:
        os.close(read_fd)
        if status is None:
            _, status = os.waitpid(pid, 0)
        untrack_worker(pid)

    rc = os.waitstatus_to_exitcode(status)
    return (
        130 if cancel_event is not None and cancel_event.is_set() else 124 if timed_out else rc
    ), timed_out
