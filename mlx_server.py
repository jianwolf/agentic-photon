"""MLX server manager for local model inference.

This module manages the mlx_lm.server subprocess lifecycle:
- Auto-starts the server with the specified model
- Waits for server readiness via health check
- Cleans up on exit

The server exposes an OpenAI-compatible API at http://127.0.0.1:{port}/v1
which can be used with PydanticAI's OpenAI provider.

Requirements:
    - macOS 15.0+ with Apple Silicon
    - mlx-lm package installed (pip install mlx-lm)
"""

import atexit
import logging
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
import threading
from typing import TextIO

logger = logging.getLogger(__name__)


class MLXServerError(Exception):
    """Raised when MLX server fails to start or respond."""
    pass


class MLXServerManager:
    """Manages mlx_lm.server subprocess lifecycle.

    Starts the MLX server on initialization and provides methods
    to check health and stop the server cleanly.

    Example:
        >>> server = MLXServerManager(model="mlx-community/Ministral-3B-Instruct-2506")
        >>> server.start()  # Blocks until ready
        >>> # Use the server at http://127.0.0.1:8080/v1
        >>> server.stop()
    """

    model: str
    port: int
    host: str
    startup_timeout: int
    watchdog_interval: int
    watchdog_failures: int
    health_timeout: float
    _process: subprocess.Popen | None
    _base_url: str
    _log_path: Path | None
    _log_file: TextIO | None
    _watchdog_thread: threading.Thread | None
    _watchdog_stop: threading.Event
    _lock: threading.Lock
    _atexit_registered: bool

    def __init__(
        self,
        model: str,
        port: int = 8080,
        host: str = "127.0.0.1",
        startup_timeout: int = 120,
        watchdog_interval: int = 30,
        watchdog_failures: int = 3,
        health_timeout: float = 2.0,
        log_path: str | Path | None = "log/mlx_server.log",
    ):
        """Initialize the server manager.

        Args:
            model: HuggingFace model ID (e.g., 'mlx-community/Ministral-3B-Instruct-2506')
            port: Port to run server on (default: 8080)
            host: Host to bind to (default: 127.0.0.1)
            startup_timeout: Max seconds to wait for server startup (default: 120)
            watchdog_interval: Seconds between health checks (0 disables watchdog)
            watchdog_failures: Consecutive failed checks before restart
            health_timeout: Timeout in seconds for health check request
            log_path: File to append MLX server logs to (default: log/mlx_server.log).
                Set to None to discard output.
        """
        self.model = model
        self.port = port
        self.host = host
        self.startup_timeout = startup_timeout
        self.watchdog_interval = watchdog_interval
        self.watchdog_failures = watchdog_failures
        self.health_timeout = health_timeout
        self._process: subprocess.Popen | None = None
        self._base_url = f"http://{host}:{port}/v1"
        self._log_path = Path(log_path) if log_path else None
        self._log_file = None
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()
        self._lock = threading.Lock()
        self._atexit_registered = False

    @property
    def base_url(self) -> str:
        """Get the server's OpenAI-compatible base URL."""
        return self._base_url

    def start(self) -> None:
        """Start the MLX server and wait for it to be ready.

        Raises:
            MLXServerError: If server fails to start or doesn't become ready
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                logger.warning("MLX server already running")
                return
            self._start_process()

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

        # Wait for server to be ready
        self._wait_for_ready()
        self._start_watchdog()

    def _start_process(self) -> None:
        cmd = [
            sys.executable, "-m", "mlx_lm.server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
        ]

        logger.info("Starting MLX server | cmd=%s", " ".join(cmd))

        try:
            stdout_target = subprocess.DEVNULL
            if self._log_path:
                stdout_target = self._ensure_log_file()
                if stdout_target is None:
                    stdout_target = subprocess.DEVNULL
            self._process = subprocess.Popen(
                cmd,
                stdout=stdout_target,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            self._close_log_file()
            raise MLXServerError(
                "mlx-lm not found. Install with: pip install mlx-lm"
            )
        except Exception as e:
            self._close_log_file()
            raise MLXServerError(f"Failed to start MLX server ({type(e).__name__}): {e}")

    def _ensure_log_file(self) -> TextIO | None:
        if not self._log_path:
            return None
        if self._log_file and not self._log_file.closed:
            return self._log_file
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", encoding="utf-8")
            return self._log_file
        except Exception as e:
            logger.warning(
                "Failed to open MLX server log file | path=%s error=%s",
                self._log_path,
                e,
            )
            self._log_file = None
            return None

    def _wait_for_ready(self, shutdown_on_timeout: bool = True) -> None:
        """Wait for server to respond to health checks.

        Raises:
            MLXServerError: If server doesn't become ready within timeout
        """
        health_url = f"http://{self.host}:{self.port}/v1/models"
        start_time = time.time()

        logger.info("Waiting for MLX server to be ready (downloading model if needed)...")

        while time.time() - start_time < self.startup_timeout:
            # Check if process died
            if self._process.poll() is not None:
                log_hint = f" See {self._log_path} for details." if self._log_path else ""
                raise MLXServerError(
                    f"MLX server exited unexpectedly (code {self._process.returncode}).{log_hint}"
                )

            # Try health check
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=self.health_timeout) as resp:
                    if resp.status == 200:
                        logger.info("MLX server ready | url=%s", self._base_url)
                        return
            except urllib.error.URLError:
                pass  # Server not ready yet
            except Exception:
                pass

            time.sleep(2)

        # Timeout reached
        if shutdown_on_timeout:
            self.stop()
        else:
            self._stop_process()
        raise MLXServerError(
            f"MLX server did not become ready within {self.startup_timeout}s. "
            "This may be due to model download. Try running manually first: "
            f"mlx_lm.server --model {self.model}"
        )

    def _check_health(self) -> bool:
        health_url = f"http://{self.host}:{self.port}/v1/models"
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=self.health_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _start_watchdog(self) -> None:
        if self.watchdog_interval <= 0 or self.watchdog_failures <= 0:
            return
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="mlx-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        failures = 0
        while not self._watchdog_stop.wait(self.watchdog_interval):
            with self._lock:
                process = self._process
            if process is None:
                continue
            if process.poll() is not None:
                failures += 1
            else:
                if self._check_health():
                    failures = 0
                    continue
                failures += 1

            if failures >= self.watchdog_failures:
                failures = 0
                logger.warning(
                    "MLX server unresponsive; restarting | interval=%ds threshold=%d",
                    self.watchdog_interval,
                    self.watchdog_failures,
                )
                self._restart_from_watchdog()

    def _restart_from_watchdog(self) -> None:
        if self._watchdog_stop.is_set():
            return
        with self._lock:
            if self._watchdog_stop.is_set():
                return
            self._stop_process()
            self._start_process()

        try:
            self._wait_for_ready(shutdown_on_timeout=False)
        except Exception as e:
            logger.warning("MLX watchdog restart failed | error=%s", e)

    def stop(self) -> None:
        """Stop the MLX server subprocess."""
        self._watchdog_stop.set()
        watchdog_thread = self._watchdog_thread
        if watchdog_thread and watchdog_thread.is_alive():
            watchdog_thread.join(timeout=5)
            if watchdog_thread.is_alive():
                logger.warning("MLX watchdog did not stop cleanly")

        with self._lock:
            if self._process is None:
                self._close_log_file()
                return

        logger.info("Stopping MLX server...")

        with self._lock:
            self._stop_process()
            self._close_log_file()

        # Unregister atexit handler
        try:
            atexit.unregister(self.stop)
            self._atexit_registered = False
        except Exception:
            pass

    def _stop_process(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("MLX server did not terminate, killing...")
                self._process.kill()
                self._process.wait()
        except Exception as e:
            logger.warning("Error stopping MLX server: %s", e)
        finally:
            self._process = None

    def _close_log_file(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self._process is not None and self._process.poll() is None
