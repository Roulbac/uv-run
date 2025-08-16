from enum import Enum, auto
import io
import logging
import os
import sys
import threading
import typing
from datetime import datetime
from contextlib import contextmanager
from rich.console import Console
from rich.text import Text

logger = logging.getLogger(__name__)


__all__ = ["prepare_subprocess_logging"]


class FDPipe:
    """A pipe for reading and writing data using standard OS file descriptors.

    This is useful if creating a subprocess with the `subprocess` module.
    """

    def __init__(self):
        super().__init__()
        self.fd_read, self.fd_write = os.pipe()
        self.reader = os.fdopen(self.fd_read, mode="rb")
        self.writer = os.fdopen(self.fd_write, mode="wb")
        self._writer_is_closed = False

    def read_is_empty(self) -> bool:
        return not self.reader.peek()

    def read_is_closed(self) -> bool:
        return self.reader.closed

    def write_is_closed(self) -> bool:
        return self._writer_is_closed

    def close_write(self):
        os.close(self.fd_write)
        self._writer_is_closed = True

    def close_read(self):
        self.reader.close()

    def get_writer(self) -> io.TextIOWrapper:
        if self.write_is_closed():
            raise ValueError(
                "Attempted to get writer while the write end of the pipe is closed."
            )
        return self.writer

    def read(self) -> bytes:
        return self.reader.readline()


class Stream(Enum):
    STDOUT = auto()
    STDERR = auto()


def _get_console(stream: Stream) -> Console:
    match stream:
        case Stream.STDOUT:
            return Console(file=sys.stdout, log_path=False, log_time=False)
        case Stream.STDERR:
            return Console(file=sys.stderr, log_path=False, log_time=False)


def _format_log_line(name: str, stream: Stream, message: str) -> str:
    line = Text()
    # Add the timestamp
    line.append(datetime.now().strftime("%X"), style="dim white")
    line.append(" - ")
    # Add the stream name
    line.append(
        stream.name.lower(),
        style="bold orange1" if stream == Stream.STDOUT else "bold blue",
    )
    line.append(" - ")
    # Add the logger's name
    line.append(name, style="dim cyan")
    line.append(" - ")
    # Add the message
    line.append(message, style="white")
    return line


class SubprocessLogger:
    """A class to stream logs from a subprocess to its parent."""

    def __init__(
        self,
        subprocess_name: str,
        stream: Stream,
    ):
        self.pipe = FDPipe()
        self.thread = threading.Thread(target=self._run, daemon=False)
        self.stream = stream
        self.pipe_read_lock = threading.Lock()
        self.subprocess_name = subprocess_name

    def _log_from_pipe(self, data: bytes):
        data_str = data.decode("utf-8").strip("\n")
        if data_str != "":
            log_line = _format_log_line(
                name=self.subprocess_name,
                stream=self.stream,
                message=data_str,
            )
            _get_console(self.stream).log(log_line)

    def _run(self):
        """Log everything that comes from the pipe."""
        while True:
            # Check if the write end is closed
            if self.pipe.write_is_closed():
                # Read until empty, then release then lock
                while data := self.pipe.read():
                    self._log_from_pipe(data)
                return
            # Check if not empty and read
            if not self.pipe.read_is_empty():
                # Acquire the lock to ensure the write end isn't closed at this time
                with self.pipe_read_lock:
                    data = self.pipe.read()
                self._log_from_pipe(data)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Close the write end of the pipe
        with self.pipe_read_lock:
            self.pipe.close_write()
        # Wait for the thread to finish and flush any remaining logs
        self.thread.join()
        # Close the read end
        self.pipe.close_read()

    def get_pipe_writer(self) -> io.TextIOWrapper:
        """Get the IO pipe writer for the subprocess logger."""
        return self.pipe.get_writer()


@contextmanager
def prepare_subprocess_logging(
    subprocess_name: str,
) -> typing.Generator[tuple[SubprocessLogger, SubprocessLogger], None, None]:
    """Prepares logging for a function ran with `subprocess`.

    Args:
        subprocess_name: The name of the subprocess to log. This is arbitrary
            but will be used to identify the subprocess in the logs in the console output.

    Returns:
        - The logger for stdout
        - The logger for stderr

    Example:
        ```
        with prepare_subprocess_logging(subprocess_name="my_subprocess") as (sl_stdout, sl_stderr):
            process = subprocess.run(
                ["echo", "Hello, world!"],
                stdout=sl_stdout.get_pipe_writer(),
                stderr=sl_stderr.get_pipe_writer(),
            )
        ```
    """
    with (
        SubprocessLogger(
            subprocess_name=subprocess_name,
            stream=Stream.STDOUT,
        ) as sl_stdout,
        SubprocessLogger(
            subprocess_name=subprocess_name,
            stream=Stream.STDERR,
        ) as sl_stderr,
    ):
        yield sl_stdout, sl_stderr
