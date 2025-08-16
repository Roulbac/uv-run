import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from queue import Queue
from textwrap import dedent
from time import sleep
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module", autouse=True)
def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )


@pytest.fixture(autouse=True)
def mock_venv_cache():
    with tempfile.TemporaryDirectory() as tempdir:
        _test_tempdir = Path(os.path.realpath(os.path.join(tempdir, "test_cache")))
        # Patch the tempfile.tempdir
        with patch("uv_func.core._get_venv_cache_dir", return_value=_test_tempdir):
            yield _test_tempdir


@pytest.fixture
def func_np_torch_src():
    return dedent("""\
        import uv_func

        @uv_func.run(dependencies=["numpy", "torch"], verbose=True)
        def inner_func():
            import torch
            import numpy as np
            return 0

        inner_func()
    """)


class TestFuncDecorator:
    @staticmethod
    def test_streams_logs(capsys):
        import uv_func

        @uv_func.run(dependencies=["numpy"], verbose=True)
        def func_foo_bar():
            import sys
            from time import sleep

            for i in range(100):
                print(f"FOO {i}")
                print(f"BAR {i}", file=sys.stderr)
                sleep(0.01)
            return 42

        # Create a background thread that captures stdout/stderr to ensure that logs are streamed
        # and NOT printed out all at once in the end
        event = threading.Event()

        stdout_logs = []
        stderr_logs = []

        def scrape_logs(
            event=event, capsys=capsys, stdout_logs=stdout_logs, stderr_logs=stderr_logs
        ):
            while not event.is_set():
                c_stdout, c_stderr = capsys.readouterr()
                stdout_logs.append(c_stdout)
                stderr_logs.append(c_stderr)
                sleep(0.01)
            # Get any remaining logs
            c_stdout, c_stderr = capsys.readouterr()
            stdout_logs.append(c_stdout)
            stderr_logs.append(c_stderr)

        thread = threading.Thread(target=scrape_logs, daemon=False)
        thread.start()

        # Run foo
        func_foo_bar()

        # Wait for the log scraping thread to finish
        event.set()
        thread.join()

        # Check that stdout and stderr logs are lists of chunks
        # This indicates the logs are streamed
        assert len(stdout_logs) > 50
        assert len(stderr_logs) > 50

        joined_stdout = "".join(stdout_logs)
        joined_stderr = "".join(stderr_logs)

        # Check that the bootstrap logs are in the stderr
        assert "numpy" in joined_stderr
        assert "cloudpickle" in joined_stderr

        # Check that the function's logs are in the right streams
        for i in range(100):
            assert f"FOO {i}" in joined_stdout
            assert f"BAR {i}" in joined_stderr

    @staticmethod
    @pytest.mark.parametrize(
        "dependencies, expected_lib_name, expected_lib_version",
        [
            (["jax[cpu]==0.7.0"], "jax", "0.7.0"),
            (["flytekit==1.16.3"], "flytekit", "1.16.3"),
            (
                ["opencv-python-headless==4.7.0.72"],
                "opencv-python-headless",
                "4.7.0.72",
            ),
        ],
    )
    def test_imports_libs(
        mock_venv_cache, dependencies, expected_lib_name, expected_lib_version
    ):
        """Test that the decorator imports the correct libraries with the correct version"""

        import uv_func

        @uv_func.run(dependencies=dependencies, verbose=True)
        def func_imports_libs():
            import importlib.metadata
            import sys

            # Get the list of installed packages
            packages_dict = {
                package.name: package.version
                for package in importlib.metadata.distributions()
            }
            return packages_dict, sys.path, sys.executable

        # Run the function
        packages_dict, sys_path, sys_executable = func_imports_libs()
        # Test for executable
        assert sys_executable.startswith(str(mock_venv_cache))
        # Test for sys.path
        assert sys_path[-1].startswith(str(mock_venv_cache))
        # Test for package version
        assert packages_dict[expected_lib_name] == expected_lib_version

    @staticmethod
    def test_does_not_hang_on_error():
        import uv_func

        @uv_func.run(dependencies=[])
        def empty_function_with_error():
            return 1 / 0

        with pytest.raises(ChildProcessError):
            empty_function_with_error()

    @staticmethod
    @pytest.mark.parametrize("daemon", [True, False])
    def test_runs_threaded(daemon):
        def threaded_func(queue):
            import uv_func

            @uv_func.run(dependencies=["numpy"])
            def func_with_venv():
                import importlib

                _ = importlib.import_module("numpy")

                return 42

            # Run the function
            res = func_with_venv()
            # Put the result in the queue
            queue.put(res)

        # Run the function in a thread
        queue = Queue()
        thread = threading.Thread(target=threaded_func, args=(queue,), daemon=daemon)
        thread.start()
        thread.join()
        # Check that the thread has finished
        assert not thread.is_alive()
        # Check that there is something in the queue
        assert not queue.empty()
        # Check that the function has returned
        assert queue.get(timeout=60) == 42

    @staticmethod
    def test_no_deadlock(func_np_torch_src):
        """Tests that two functions using the same pip dependencies do not deadlock"""
        for _ in range(10):
            p1 = subprocess.Popen(
                [sys.executable, "-c", f"{func_np_torch_src}"],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            p2 = subprocess.Popen(
                [sys.executable, "-c", f"{func_np_torch_src}"],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            # Wait for the two processes to finish
            p1.wait()
            p2.wait()
            # Check that the processes have finished successfully
            assert p1.returncode == 0 and p2.returncode == 0
