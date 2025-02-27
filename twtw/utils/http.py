"""
HTTP utilities for TWTW.
"""
import asyncio
import time
from collections import defaultdict

# Rate limiting configuration
RATE_LIMIT = 1  # requests per second per domain
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 30  # seconds

class RateLimiter:
    """
    Rate limiter to prevent overwhelming source websites with too many requests.
    """
    def __init__(self):
        self.last_requests = defaultdict(lambda: 0.0)
        self.locks = defaultdict(asyncio.Lock)

    async def acquire(self, domain: str):
        """
        Acquire rate limit for a domain.
        
        Args:
            domain: The domain to rate limit
        """
        async with self.locks[domain]:
            now = time.time()
            time_passed = now - self.last_requests[domain]
            if time_passed < RATE_LIMIT:
                await asyncio.sleep(RATE_LIMIT - time_passed)
            self.last_requests[domain] = time.time() 