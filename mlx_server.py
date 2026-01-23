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

    def __init__(
        self,
        model: str,
        port: int = 8080,
        host: str = "127.0.0.1",
        startup_timeout: int = 120,
    ):
        """Initialize the server manager.

        Args:
            model: HuggingFace model ID (e.g., 'mlx-community/Ministral-3B-Instruct-2506')
            port: Port to run server on (default: 8080)
            host: Host to bind to (default: 127.0.0.1)
            startup_timeout: Max seconds to wait for server startup (default: 120)
        """
        self.model = model
        self.port = port
        self.host = host
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None
        self._base_url = f"http://{host}:{port}/v1"

    @property
    def base_url(self) -> str:
        """Get the server's OpenAI-compatible base URL."""
        return self._base_url

    def start(self) -> None:
        """Start the MLX server and wait for it to be ready.

        Raises:
            MLXServerError: If server fails to start or doesn't become ready
        """
        if self._process is not None:
            logger.warning("MLX server already running")
            return

        cmd = [
            sys.executable, "-m", "mlx_lm.server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
        ]

        logger.info("Starting MLX server | cmd=%s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            raise MLXServerError(
                "mlx-lm not found. Install with: pip install mlx-lm"
            )
        except Exception as e:
            raise MLXServerError(f"Failed to start MLX server ({type(e).__name__}): {e}")

        # Register cleanup on exit
        atexit.register(self.stop)

        # Wait for server to be ready
        self._wait_for_ready()

    def _wait_for_ready(self) -> None:
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
                # Process exited, read output for error message
                stdout, _ = self._process.communicate()
                raise MLXServerError(
                    f"MLX server exited unexpectedly (code {self._process.returncode}):\n{stdout}"
                )

            # Try health check
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("MLX server ready | url=%s", self._base_url)
                        return
            except urllib.error.URLError:
                pass  # Server not ready yet
            except Exception:
                pass

            time.sleep(2)

        # Timeout reached
        self.stop()
        raise MLXServerError(
            f"MLX server did not become ready within {self.startup_timeout}s. "
            "This may be due to model download. Try running manually first: "
            f"mlx_lm.server --model {self.model}"
        )

    def stop(self) -> None:
        """Stop the MLX server subprocess."""
        if self._process is None:
            return

        logger.info("Stopping MLX server...")

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

        # Unregister atexit handler
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass

    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self._process is not None and self._process.poll() is None
