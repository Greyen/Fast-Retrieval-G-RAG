
# Define a direct print function for critical logs that must be visible in all processes
def direct_log(message, level="INFO", enable_output: bool = True):
    """
    Log a message directly to stderr to ensure visibility in all processes,
    including the Gunicorn master process.

    Args:
        message: The message to log
        level: Log level (default: "INFO")
        enable_output: Whether to actually output the log (default: True)
    """
    if enable_output:
        print(f"{level}: {message}", file=sys.stderr, flush=True)
        
_is_multiprocess = None


# locks for mutex access
_storage_lock: Optional[LockType] = None
_internal_lock: Optional[LockType] = None

# async locks for coroutine synchronization in multiprocess mode
_async_locks: Optional[Dict[str, asyncio.Lock]] = None

class UnifiedLock(Generic[T]):
    """Provide a unified lock interface type for asyncio.Lock and multiprocessing.Lock"""

    def __init__(
        self,
        lock: Union[ProcessLock, asyncio.Lock],
        is_async: bool,
        name: str = "unnamed",
        enable_logging: bool = True,
        async_lock: Optional[asyncio.Lock] = None,
    ):
        self._lock = lock
        self._is_async = is_async
        self._pid = os.getpid()  # for debug only
        self._name = name  # for debug only
        self._enable_logging = enable_logging  # for debug only
        self._async_lock = async_lock  # auxiliary lock for coroutine synchronization

    async def __aenter__(self) -> "UnifiedLock[T]":
        try:
            # direct_log(
            #     f"== Lock == Process {self._pid}: Acquiring lock '{self._name}' (async={self._is_async})",
            #     enable_output=self._enable_logging,
            # )

            # If in multiprocess mode and async lock exists, acquire it first
            if not self._is_async and self._async_lock is not None:
                # direct_log(
                #     f"== Lock == Process {self._pid}: Acquiring async lock for '{self._name}'",
                #     enable_output=self._enable_logging,
                # )
                await self._async_lock.acquire()
                direct_log(
                    f"== Lock == Process {self._pid}: Async lock for '{self._name}' acquired",
                    enable_output=self._enable_logging,
                )

            # Then acquire the main lock
            if self._is_async:
                await self._lock.acquire()
            else:
                self._lock.acquire()

            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' acquired (async={self._is_async})",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            # If main lock acquisition fails, release the async lock if it was acquired
            if (
                not self._is_async
                and self._async_lock is not None
                and self._async_lock.locked()
            ):
                self._async_lock.release()

            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}': {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        main_lock_released = False
        try:
            # Release main lock first
            if self._is_async:
                self._lock.release()
            else:
                self._lock.release()
            main_lock_released = True

            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' released (async={self._is_async})",
                enable_output=self._enable_logging,
            )

            # Then release async lock if in multiprocess mode
            if not self._is_async and self._async_lock is not None:
                self._async_lock.release()
                direct_log(
                    f"== Lock == Process {self._pid}: Async lock '{self._name}' released",
                    enable_output=self._enable_logging,
                )

        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}': {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )

            # If main lock release failed but async lock hasn't been released, try to release it
            if (
                not main_lock_released
                and not self._is_async
                and self._async_lock is not None
            ):
                try:
                    direct_log(
                        f"== Lock == Process {self._pid}: Attempting to release async lock after main lock failure",
                        level="WARNING",
                        enable_output=self._enable_logging,
                    )
                    self._async_lock.release()
                    direct_log(
                        f"== Lock == Process {self._pid}: Successfully released async lock after main lock failure",
                        enable_output=self._enable_logging,
                    )
                except Exception as inner_e:
                    direct_log(
                        f"== Lock == Process {self._pid}: Failed to release async lock after main lock failure: {inner_e}",
                        level="ERROR",
                        enable_output=self._enable_logging,
                    )

            raise

    def __enter__(self) -> "UnifiedLock[T]":
        """For backward compatibility"""
        try:
            if self._is_async:
                raise RuntimeError("Use 'async with' for shared_storage lock")
            direct_log(
                f"== Lock == Process {self._pid}: Acquiring lock '{self._name}' (sync)",
                enable_output=self._enable_logging,
            )
            self._lock.acquire()
            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' acquired (sync)",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For backward compatibility"""
        try:
            if self._is_async:
                raise RuntimeError("Use 'async with' for shared_storage lock")
            direct_log(
                f"== Lock == Process {self._pid}: Releasing lock '{self._name}' (sync)",
                enable_output=self._enable_logging,
            )
            self._lock.release()
            direct_log(
                f"== Lock == Process {self._pid}: Lock '{self._name}' released (sync)",
                enable_output=self._enable_logging,
            )
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=self._enable_logging,
            )
            raise



def get_internal_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified storage lock for data consistency"""
    async_lock = _async_locks.get("internal_lock") if _is_multiprocess else None
    return UnifiedLock(
        lock=_internal_lock,
        is_async=not _is_multiprocess,
        name="internal_lock",
        enable_logging=enable_logging,
        async_lock=async_lock,
    )

async def get_namespace_data(namespace: str) -> Dict[str, Any]:
    """get the shared data reference for specific namespace"""
    if _shared_dicts is None:
        direct_log(
            f"Error: try to getnanmespace before it is initialized, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Shared dictionaries not initialized")

    async with get_internal_lock():
        if namespace not in _shared_dicts:
            if _is_multiprocess and _manager is not None:
                _shared_dicts[namespace] = _manager.dict()
            else:
                _shared_dicts[namespace] = {}

    return _shared_dicts[namespace]


async def initialize_pipeline_status():
    """
    Initialize pipeline namespace with default values.
    This function is called during FASTAPI lifespan for each worker.
    """
