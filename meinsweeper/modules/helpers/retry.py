import asyncio
import time
from typing import Callable, TypeVar, Optional
from functools import wraps

T = TypeVar('T')

class RetryStrategy:
    """Handles retry logic with exponential backoff"""
    def __init__(
        self, 
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        logger = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.logger = logger

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> tuple[bool, Optional[T], Optional[Exception]]:
        """
        Execute function with retry logic
        Returns: (success, result, last_error)
        """
        retry_count = 0
        last_error = None
        current_delay = self.initial_delay

        while retry_count < self.max_retries:
            try:
                result = await func(*args, **kwargs)
                return True, result, None
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count < self.max_retries:
                    if self.logger:
                        self.logger.debug(
                            f"Attempt {retry_count} failed: {str(e)}. "
                            f"Retrying in {current_delay:.1f}s"
                        )
                    await asyncio.sleep(current_delay)
                    current_delay = min(
                        current_delay * self.backoff_factor,
                        self.max_delay
                    )
                else:
                    if self.logger:
                        self.logger.warning(
                            f"All {self.max_retries} attempts failed. Last error: {str(e)}"
                        )

        return False, None, last_error 