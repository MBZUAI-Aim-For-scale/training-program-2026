"""Separate user-requested cancellation from computation and cleanup failures."""


def cancellation_warning(last_run):
    warnings = []
    for result in last_run.get("runs", []):
        warning = result.get("stop_warning")
        if warning:
            warnings.append(warning)
        elif "PermissionError" in (result.get("stderr") or ""):
            warnings.append("The operating system denied a cleanup operation. Some workers may still be running; verify they have exited before starting another benchmark.")
    return " ".join(dict.fromkeys(warnings))


def execution_status(result, cancelled=False):
    if result.get("ok"):
        return "Completed"
    if result.get("cancelled") or cancelled:
        return "Stop requested — cleanup not confirmed" if cancellation_warning({"runs": [result]}) else "Cancelled by you"
    if result.get('returncode') == -11:
        return 'Failed — segmentation fault (SIGSEGV, exit code -11)'
    if result.get("timed_out") or result.get("returncode") == 124:
        seconds = result.get("timeout_seconds")
        limit = f" after {seconds} seconds" if seconds is not None else ""
        return f"Timed out{limit} — increase the benchmark time limit or reduce the evaluation period"
    return f"Failed — exit code {result.get('returncode', 'unknown')}"
