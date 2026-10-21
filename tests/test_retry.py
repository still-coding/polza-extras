import random

import pytest

from polza_extras.errors import error_from_response
from polza_extras.retry import RetryPolicy


def err(status, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return error_from_response(status, {"error": {"code": "X", "message": "m"}}, headers)


policy = RetryPolicy(max_retries=3, base_delay=1.0, max_delay=30.0, jitter=0.0)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 409])
def test_never_retried(status):
    assert policy.delay_for(err(status), attempt=0) is None


@pytest.mark.parametrize("status", [408, 500, 502])
def test_retried_after_exponential_pause(status):
    assert [policy.delay_for(err(status), a) for a in range(3)] == [1.0, 2.0, 4.0]


def test_gives_up_after_max_retries():
    assert policy.delay_for(err(502), attempt=3) is None


def test_503_honors_retry_after_else_backoff():
    assert policy.delay_for(err(503, retry_after=7), 0) == 7.0
    assert policy.delay_for(err(503), 1) == 2.0


def test_429_own_limit_waits_retry_after():
    assert policy.delay_for(err(429, retry_after=12), 0) == 12.0


def test_429_provider_limit_backs_off_1_2_4_8():
    p = RetryPolicy(max_retries=4, base_delay=1.0, jitter=0.0)
    assert [p.delay_for(err(429), a) for a in range(4)] == [1.0, 2.0, 4.0, 8.0]


def test_429_wait_longer_than_max_wait_is_not_retried():
    # "own" limit that resets at midnight Moscow time: fail fast instead of sleeping for hours
    p = RetryPolicy(max_wait=60.0, jitter=0.0)
    assert p.delay_for(err(429, retry_after=3600), 0) is None


def test_402_not_retried_by_default_even_under_load():
    assert policy.delay_for(err(402), 0) is None


def test_402_opt_in_retries_briefly_twice():
    p = RetryPolicy(retry_402=True, jitter=0.0)
    assert [p.delay_for(err(402), a) for a in range(3)] == [1.0, 1.0, None]


def test_jitter_stays_within_bounds_and_is_seedable():
    p = RetryPolicy(base_delay=1.0, jitter=0.25)
    rng = random.Random(7)
    delays = [p.delay_for(err(502), 0, rng=rng) for _ in range(200)]
    assert all(0.75 <= d <= 1.25 for d in delays) and len(set(delays)) > 1
