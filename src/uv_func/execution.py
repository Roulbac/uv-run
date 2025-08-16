import os
import subprocess
from string import Template
from typing import Callable, ParamSpec, TypeVar

import cloudpickle
from rich.traceback import Traceback

from .logging import Stream, _get_console, prepare_subprocess_logging

P = ParamSpec("P")
R = TypeVar("R")

_CHILD_PROCESS_ENV_VAR = "uv_func_CHILD_PROCESS"


def is_in_child_process() -> bool:
    return os.environ.get(_CHILD_PROCESS_ENV_VAR, "0") == "1"


_SCRIPT_TEMPLATE = Template("""\
import os
import sys
import cloudpickle
import traceback
from rich.traceback import Traceback

os.environ["${CHILD_PROCESS_ENV_VAR}"] = "1"

out_file = os.fdopen(${OUT_WFD}, 'wb')
try:
    # Read everything as a single dictionary from the provided FD
    with os.fdopen(${IN_RFD}, 'rb') as in_file:
        data = cloudpickle.loads(in_file.read())
    
    # Extract components from the dictionary and deserialize them again
    func = cloudpickle.loads(data['func'])
    args = cloudpickle.loads(data['args'])
    kwargs = cloudpickle.loads(data['kwargs'])
    # Call func with args and kwargs
    result = func(*args, **kwargs)
    # Serialize the result and send it back to the parent process
    out_file.write(cloudpickle.dumps({"result": result, "success": True}))
except Exception as e:
    # Handle exceptions and return them in a structured way
    error_data = {
        "success": False,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc(),
        "rich_traceback": Traceback.from_exception(type(e), e, e.__traceback__)
    }
    out_file.write(cloudpickle.dumps(error_data))
finally:
    out_file.close()
""")


class CloudPickleFuncWrapper:
    """Wrapper that serializes a function and its arguments using cloudpickle and executes it in a child process."""

    def __init__(
        self,
        func: Callable[P, R],
        venv_executable: str,
    ):
        self._func = func
        self._func_serialized = cloudpickle.dumps(func)
        self._venv_executable = venv_executable

    def _serialize_objects(self, *args: P.args, **kwargs: P.kwargs) -> bytes:
        """Serializes args/kwargs to be sent to the child process for execution with the target function."""
        return cloudpickle.dumps(
            {
                "func": self._func_serialized,
                "args": cloudpickle.dumps(args),
                "kwargs": cloudpickle.dumps(kwargs),
            }
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        # Prepare data as a single dictionary to avoid race conditions
        data = self._serialize_objects(*args, **kwargs)
        # Create two pipes: input write/read and output write/read
        in_rfd, in_wfd = os.pipe()
        out_rfd, out_wfd = os.pipe()
        # Prepare the child script by substituting FDs
        script = _SCRIPT_TEMPLATE.substitute(
            IN_RFD=str(in_rfd),
            OUT_WFD=str(out_wfd),
            CHILD_PROCESS_ENV_VAR=_CHILD_PROCESS_ENV_VAR,
        )
        try:
            # Launch the child process with the substituted script
            # Pass only the pipe ends needed by the child
            with prepare_subprocess_logging(
                subprocess_name=self._func.__name__,
            ) as (sl_stdout, sl_stderr):
                # Launch the child process with the substituted script
                proc = subprocess.Popen(
                    [self._venv_executable, "-c", script],
                    stdout=sl_stdout.get_pipe_writer(),
                    stderr=sl_stderr.get_pipe_writer(),
                    pass_fds=[in_rfd, out_wfd],
                )
                # Send serialized data through the input pipe
                # This will contain the function, its args and its kwargs
                with os.fdopen(in_wfd, "wb") as in_fp:
                    in_fp.write(data)
                # Wait for the process to complete
                proc.wait()
                # Make sure the out write FD is closed before reading from the out read FD
                os.close(out_wfd)
                # Read response from the output pipe (now will see EOF when child closes its end)
                with os.fdopen(out_rfd, "rb") as out_fp:
                    response_bytes = out_fp.read()

                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, proc.args)

                # Deserialize the response
                response = cloudpickle.loads(response_bytes)

                if response["success"]:
                    return response["result"]
                else:
                    tb: Traceback = response["rich_traceback"]
                    console = _get_console(Stream.STDERR)
                    # Print the traceback from the child process
                    console.rule("[bold red]CHILD PROCESS TRACEBACK[/]", style="red")
                    console.print(tb)
                    console.rule(
                        "[bold red]END CHILD PROCESS TRACEBACK[/]", style="red"
                    )
                    # Re-raise the error to be caught by the parent process
                    raise ChildProcessError(
                        f"Subprocess execution of '{self._func.__name__}' failed with error '{response['error_type']}' and message '{response['error_message']}'. Please see the traceback above for more details."
                    )
        finally:
            # Ensure any remaining FDs are closed
            for fd in (in_rfd, in_wfd, out_rfd, out_wfd):
                try:
                    os.close(fd)
                except OSError:
                    pass  # Already closed
