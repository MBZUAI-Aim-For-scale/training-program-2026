"""Bind a refreshed browser session to its existing server-side benchmark."""

from ai_weather.benchmark_registry import find_benchmark_job, register_benchmark_job


def reconnect_benchmark(state, query_params):
    active = state.get('benchmark_active_job')
    if active is not None:
        # Also make jobs from a still-open, older page refresh-safe on rerun.
        if not state.get('benchmark_job_token'):
            remember_benchmark(state, query_params, active)
        return
    token = query_params.get('benchmark_job')
    if not token or token == state.get('benchmark_job_token'):
        return
    state['benchmark_job_token'] = token
    job = find_benchmark_job(token)
    if job is not None:
        state['benchmark_active_job'] = job
    else:
        state['benchmark_reconnect_error'] = (
            'The previous benchmark is no longer available on this server. '
            'It may have restarted, or the saved job has expired.')


def remember_benchmark(state, query_params, job):
    token = register_benchmark_job(job)
    state['benchmark_active_job'] = job
    state['benchmark_job_token'] = token
    state.pop('benchmark_reconnect_error', None)
    query_params['benchmark_job'] = token
