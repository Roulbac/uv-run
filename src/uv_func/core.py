import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
from functools import cache, wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import portalocker

from .execution import CloudPickleFuncWrapper, is_in_child_process
from .logging import prepare_subprocess_logging

__all__ = ["run"]

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


CACHE_DIR = Path.home() / ".cache" / "uv_func"


def _get_venv_cache_dir() -> Path:
    return CACHE_DIR / "venvs"


def _get_venv_executable_path(venv_path: os.PathLike | str) -> os.PathLike | str:
    """Get the path to the python executable of the virtual environment."""
    # Find the python in the bin directory
    return (Path(venv_path) / "bin" / "python").absolute()


def _get_venv_lock_path(venv_path: os.PathLike | str) -> os.PathLike | str:
    """Get the path to the lock of the virtual environment."""
    locks_base_path = _get_venv_cache_dir() / "locks"
    if not os.path.exists(locks_base_path):
        os.makedirs(locks_base_path, exist_ok=True)
    return locks_base_path / f"{os.path.basename(venv_path)}.lock"


@cache
def _check_uv_installed():
    if not shutil.which("uv"):
        raise RuntimeError(
            "The 'uv' executable is not in the PATH. Please install it following the instructions at https://docs.astral.sh/uv/getting-started/installation/."
        )


def _setup(
    venv_path: os.PathLike,
    dependencies: list[str],
    func_name: str,
    verbose: bool = False,
):
    lock_path = _get_venv_lock_path(venv_path)
    logger.debug(f"Attempting to acquire lock for virtual environment: {venv_path}")
    # Acquire a lock on the virtual environment to ensure no other process is using it
    with portalocker.Lock(
        lock_path, mode="ab", fail_when_locked=False, flags=portalocker.LOCK_EX
    ):
        logger.debug(f"Successfully acquired lock for virtual environment: {venv_path}")
        done_marker_path = os.path.realpath(os.path.join(venv_path, ".done_marker"))
        if not os.path.exists(done_marker_path):
            logger.info(
                f"No bootstrapped virtual environment found at {venv_path}. Creating a new one..."
            )
            shutil.rmtree(venv_path, ignore_errors=True)
            python_version = sys.version_info[:3]
            _check_uv_installed()
            subprocess.run(
                [
                    "uv",
                    "-q",
                    "venv",
                    "--python",
                    ".".join(map(str, python_version)),
                    venv_path,
                ]
            )
        venv_executable = _get_venv_executable_path(venv_path)
        command = ["uv", "pip", "install", "--python", venv_executable]
        if not verbose:
            command += ["-q", "-q"]
        command += list(dependencies)
        # Check that uv is installed before running the command
        _check_uv_installed()
        with prepare_subprocess_logging(subprocess_name=f"{func_name}_setup") as (
            sl_stdout,
            sl_stderr,
        ):
            proc = subprocess.Popen(
                command,
                stdout=sl_stdout.get_pipe_writer(),
                stderr=sl_stderr.get_pipe_writer(),
            )
            proc.wait()

        if proc.returncode != 0:
            if os.path.exists(done_marker_path):
                os.remove(done_marker_path)
            shutil.rmtree(venv_path, ignore_errors=True)
            logger.error(
                f"Failed to install dependencies in {venv_path}. The environment has been cleaned up. Dependencies: {dependencies}"
            )
            raise ChildProcessError(
                f"Failed to install dependencies in {venv_path}. The environment has been cleaned up. Dependencies: {dependencies}"
            )

        with open(done_marker_path, "w") as f:
            f.write("")


def run(
    dependencies: list[str], verbose: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator that allows a function to be executed in a virtual environment.

    Args:
        dependencies: A list of pip dependencies to install into the virtual environment.
        verbose: Whether log the venv setup process to the console or not.

    Example usage:
    ```python
    import uv_func

    @uv_func.run(dependencies=["torch==2.0.1", "torchvision==0.15.2"])
    def my_function():
        import torch
        return torch.tensor(42).item()

    x = my_function()
    assert x == 42
    ```

    Raises:
        - RuntimeError: If the 'uv' executable is not in the PATH.
        - ChildProcessError: If the pip install fails or the function execution fails within its isolated environment.

    """

    def decorator(
        func: Callable[P, R], *, dependencies=dependencies, verbose=verbose
    ) -> Callable[P, R]:
        # If we are within the function's child process, do not try to re-bootstrap the venv
        if is_in_child_process():
            return func

        packages_dict = {
            package.name: package for package in importlib.metadata.distributions()
        }

        def _get_uv_func_pip_spec(dist=packages_dict["uv-func"]) -> list[str]:
            if direct_url_json := dist.read_text("direct_url.json"):
                src = json.loads(direct_url_json)
                return f"uv-func @ {src['url']}"
            else:
                return f"uv-func=={dist.version}"

        dependencies = sorted(
            [
                # Pre-pend uv-func and cloudpickle as mandatory pip dependencies
                # to the venv where the function is executed
                f"cloudpickle=={packages_dict['cloudpickle'].version}",
                f"rich=={packages_dict['rich'].version}",
                _get_uv_func_pip_spec(),
                *dependencies,
            ]
        )
        venv_cache_dir = _get_venv_cache_dir()
        os.makedirs(venv_cache_dir, exist_ok=True)
        venv_cache_dir = os.path.realpath(venv_cache_dir)
        # Generate a path-safe hash of the pip dependencies
        # this prevents the duplication of virtual environments
        deps_hash = hashlib.md5("".join(sorted(dependencies)).encode()).hexdigest()[:8]
        venv_path = os.path.join(venv_cache_dir, f"{deps_hash}")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Wait for the pip install to finish
            logger.debug(
                f"Preparing isolated environment for {func.__name__} at {venv_path}..."
            )

            _setup(venv_path, dependencies, func.__name__, verbose)
            venv_executable = _get_venv_executable_path(venv_path)

            logger.debug(
                f"Environment ready. Running {func.__name__} in a new process..."
            )
            func_wrapper = CloudPickleFuncWrapper(func, venv_executable)
            result = func_wrapper(*args, **kwargs)
            return result

        return wrapper

    return decorator
