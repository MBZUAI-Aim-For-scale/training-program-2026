# ui/utils/concurrent_processor.py
from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_WORKERS = 4
# Ceiling on how long the whole batch will wait. Without this, one stuck
# task (a hung network call, a file read against an unresponsive mount,
# etc.) blocks every other result — and since nothing renders until
# run_concurrent_tasks returns, that freezes the whole page. This is the
# same failure mode a bare `with ThreadPoolExecutor(...) as executor:`
# has: its implicit shutdown(wait=True) blocks until *every* task
# finishes, timeout or not.
DEFAULT_TIMEOUT = 25

# Worker threads have no Streamlit ScriptRunContext by default. Tasks that
# call st.cache_data/st.cache_resource internally (fine — that's just data
# caching) still need a context attached or those calls can misbehave.
# Actual UI rendering (st.markdown, st.image, widgets, ...) must NOT happen
# inside a submitted task regardless — render sequentially afterwards, in
# the main thread, using the results this function returns.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:  # pragma: no cover - older Streamlit versions
    try:
        from streamlit.script_run_context import add_script_run_ctx, get_script_run_ctx  # type: ignore
    except ImportError:
        add_script_run_ctx = None
        get_script_run_ctx = None


def run_concurrent_tasks(
    tasks: List[Callable],
    args_list: Optional[List[Tuple]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_workers: int = MAX_WORKERS,
) -> List[Any]:
    """
    Execute multiple functions in parallel, safely, from inside a Streamlit
    script run.

    - Tasks must not touch Streamlit UI calls (st.markdown/st.image/widgets/
      etc.) — only fetch or compute data. Render results afterwards in the
      main thread.
    - The whole batch is bounded by `timeout` seconds. A task that doesn't
      finish in time does NOT block the others or delay this function's
      return past `timeout`; its slot in the result list holds a
      TimeoutError instead, for the caller to handle (e.g. show a warning
      for that one section instead of freezing the page).
    - Returns a list the same length as `tasks`. Each entry is either the
      task's return value, or an Exception instance if that task raised or
      timed out — check with `isinstance(result, Exception)`.
    """
    if args_list is None:
        args_list = [()] * len(tasks)
    results: List[Any] = [None] * len(tasks)
    if not tasks:
        return results

    ctx = get_script_run_ctx() if get_script_run_ctx else None

    def _run(fn, args):
        if ctx is not None and add_script_run_ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        return fn(*args)

    # Deliberately NOT `with ThreadPoolExecutor(...) as executor:` — that
    # context manager's implicit shutdown(wait=True) blocks on exit until
    # *every* submitted task finishes, including ones we've already given
    # up on below. That silently defeats the timeout and freezes the page
    # again on any hung task.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_to_idx = {
        executor.submit(_run, task, args): idx
        for idx, (task, args) in enumerate(zip(tasks, args_list))
    }
    try:
        # as_completed's own `timeout` bounds the *whole wait*: if it
        # elapses before every future is done, it raises TimeoutError from
        # the iterator itself rather than hanging on whichever task is
        # slowest.
        for future in concurrent.futures.as_completed(future_to_idx, timeout=timeout):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Log only — do NOT call st.error/st.warning inside a
                # thread. Return the exception so the caller renders the
                # failure state safely on the main thread.
                logger.error(f"Background thread task {idx} failed: {e}", exc_info=True)
                results[idx] = e
    except concurrent.futures.TimeoutError:
        pending = [idx for idx in future_to_idx.values() if results[idx] is None]
        for idx in pending:
            logger.error(f"Background thread task {idx} did not complete within {timeout}s")
            results[idx] = TimeoutError(f"Task {idx} timed out after {timeout}s")
    finally:
        # wait=False: return control to the page now. Any task still
        # running past its timeout (e.g. a socket read with no underlying
        # timeout, or a slow disk) keeps running in the background and
        # exits on its own rather than blocking this call or the app.
        executor.shutdown(wait=False)

    return results