from __future__ import annotations

import random
from dataclasses import dataclass

from .errors import PolzaError, RateLimitError

_BACKOFF_STATUSES = frozenset({0, 408, 500, 502})


@dataclass(frozen=True)
class RetryPolicy:
    """Retry rules from the Polza docs.

    408/500/502: retry after a pause. 503: honor Retry-After, else back off.
    429: wait Retry-After (own limit) or back off (provider limit). 402 is retried only when
    `retry_402` is set, because Polza gives no machine-readable way to tell a parallel-load
    reserve conflict from an empty balance. 400/401/403/404/409/413 are never retried.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    max_wait: float = 60.0
    jitter: float = 0.25
    retry_402: bool = False
    retry_402_attempts: int = 2

    def delay_for(
        self, error: PolzaError, attempt: int, rng: random.Random | None = None
    ) -> float | None:
        """Seconds to sleep before retry number `attempt` (0-based), or None to give up."""
        if attempt >= self.max_retries:
            return None
        status = error.status
        if status == 402:
            if not self.retry_402 or attempt >= self.retry_402_attempts:
                return None
            return self._jittered(self.base_delay, rng)
        if isinstance(error, RateLimitError) and error.retry_after is not None:
            return error.retry_after if error.retry_after <= self.max_wait else None
        if status == 503 and error.retry_after is not None:
            return error.retry_after
        if status in _BACKOFF_STATUSES or status in (429, 503):
            return self._jittered(min(self.base_delay * 2**attempt, self.max_delay), rng)
        return None

    def _jittered(self, delay: float, rng: random.Random | None) -> float:
        if not self.jitter:
            return delay
        return delay * (1 + (rng or random).uniform(-self.jitter, self.jitter))
