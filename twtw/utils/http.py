"""
HTTP utilities for TWTW.
"""
import asyncio
import time
import logging
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT = 1  # requests per second per domain
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 60  # seconds - Increased from 30 to 60

class RateLimiter:
    """
    Rate limiter to prevent overwhelming source websites with too many requests.
    Includes enhanced error handling and recovery mechanisms.
    """
    def __init__(self):
        self.last_requests = defaultdict(lambda: 0.0)
        self.locks = defaultdict(asyncio.Lock)
        self.failure_counts = defaultdict(int)
        self.backoff_times = defaultdict(lambda: 1.0)  # Start with 1 second backoff
        self.max_backoff = 60.0  # Maximum backoff in seconds
        self.failure_threshold = 3  # Number of failures before increasing backoff

    async def acquire(self, domain: str):
        """
        Acquire rate limit for a domain with adaptive backoff for failing domains.
        
        Args:
            domain: The domain to rate limit
        """
        try:
            async with self.locks[domain]:
                now = time.time()
                time_passed = now - self.last_requests[domain]
                
                # Calculate required wait time
                wait_time = max(RATE_LIMIT, self.backoff_times[domain]) - time_passed
                
                if wait_time > 0:
                    logger.debug(f"Rate limiting {domain}, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    
                self.last_requests[domain] = time.time()
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.warning(f"Rate limiter acquisition for {domain} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in rate limiter for {domain}: {e}")
            # Still update the last request time to prevent hammering in case of errors
            self.last_requests[domain] = time.time()
            raise

    def report_success(self, domain: str):
        """
        Report a successful request to a domain.
        This will gradually reduce the backoff time for the domain.
        
        Args:
            domain: The domain that had a successful request
        """
        self.failure_counts[domain] = 0
        # Gradually reduce backoff time, but never below the base rate limit
        if self.backoff_times[domain] > RATE_LIMIT:
            self.backoff_times[domain] = max(RATE_LIMIT, self.backoff_times[domain] * 0.8)

    def report_failure(self, domain: str):
        """
        Report a failed request to a domain.
        This will increase the backoff time for the domain.
        
        Args:
            domain: The domain that had a failed request
        """
        self.failure_counts[domain] += 1
        
        # Increase backoff time after reaching threshold
        if self.failure_counts[domain] >= self.failure_threshold:
            self.backoff_times[domain] = min(self.max_backoff, self.backoff_times[domain] * 2.0)
            logger.warning(
                f"Increased backoff for {domain} to {self.backoff_times[domain]:.2f}s "
                f"after {self.failure_counts[domain]} failures"
            ) 