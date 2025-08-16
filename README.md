# uv-func

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Run Python functions in isolated UV virtual environments with their own dependencies – like `uv run` for individual functions.

## Why uv-func?

**Problem:** Need different library versions in the same program? Want to use heavy dependencies (PyTorch, TensorFlow) in just one function? Dealing with dependency conflicts?

**Solution:** Decorate functions to run in isolated environments with specific dependencies, automatic caching, and real-time logging.

## Quick Start

### Installation

```bash
pip install uv-func
```

### Usage

```python
import uv_func

@uv_func.run(dependencies=["torch==2.8.0", "numpy==1.26.4"])
def torch_sum(x: list[int], y: list[int]) -> list[float]:
    import torch
    x_tensor = torch.tensor(x).float()
    y_tensor = torch.tensor(y).float()
    return (x_tensor + y_tensor).tolist()  # Return serializable list

result = torch_sum([1,2,3], [4,5,6])  # [5.0, 7.0, 9.0]
```

## Examples

### Different Library Versions
```python
@uv_func.run(dependencies=["numpy==1.21.0"])
def legacy_compute(data: list) -> float:
    import numpy as np
    return float(np.array(data).sum())

@uv_func.run(dependencies=["numpy==1.26.4"])
def modern_compute(data: list) -> float:
    import numpy as np
    return float(np.array(data, dtype=np.float32).mean())
```

### Data Processing Pipeline
```python
@uv_func.run(dependencies=["polars==0.20.0"])
def process_data(file_path: str) -> dict:
    import polars as pl
    df = pl.read_csv(file_path).filter(pl.col("value") > 100)
    return {"data": df.to_dicts(), "shape": df.shape}

@uv_func.run(dependencies=["matplotlib==3.8.0", "seaborn==0.13.0"])
def visualize(data: list[float], filename: str = "plot.png") -> dict:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.histplot(data)
    plt.savefig(filename)
    plt.close()
    return {"filename": filename, "points": len(data)}
```

## How It Works

1. **Hash dependencies** → Check cache for existing environment
2. **Create/reuse** UV virtual environment with exact dependencies
3. **Serialize** function + arguments with `cloudpickle`
4. **Execute** in subprocess with isolated Python interpreter
5. **Return** serialized results to parent process

## ⚠️ Serialization Constraints

**uv-func uses `cloudpickle` for IPC. Keep inputs/outputs simple!**

✅ **Use:** primitives (`int`, `float`, `str`, `bool`), collections (`list`, `dict`, `tuple`), simple dataclasses  
❌ **Avoid:** complex objects, file handles, database connections, threading locks, raw library objects

```python
# ✅ GOOD: Return dict
@uv_func.run(dependencies=["requests"])
def fetch(url: str) -> dict:
    import requests
    resp = requests.get(url)
    return {"status": resp.status_code, "data": resp.json()}

# ❌ BAD: Return complex object
@uv_func.run(dependencies=["requests"])
def fetch_bad(url: str):  # Don't do this
    import requests
    return requests.get(url)  # Response object won't serialize
```

## API Reference

### `@uv_func.run(dependencies, verbose=False)`

- **dependencies** (`list[str]`): Pip-style dependencies (e.g., `["torch==2.0.1", "numpy>=1.20"]`)
- **verbose** (`bool`): Enable detailed environment setup logs
- **Returns**: Callable executing function in isolation

**Note:** Arguments and returns must be cloudpickle-serializable.

## Features

- 🚀 **Fast**: Leverages UV's speed for environment management
- 📦 **Cached**: Reuses environments based on dependency hash
- 🔄 **Streaming**: Real-time stdout/stderr from isolated functions
- 🛡️ **Safe**: Full exception propagation with tracebacks
- 🧵 **Thread-safe**: Works in multi-threaded applications

## Advanced Usage

```python
# Verbose mode for debugging
@uv_func.run(dependencies=["tensorflow==2.13.0"], verbose=True)
def train():
    import tensorflow as tf
    return tf.__version__

# Error handling
try:
    result = some_isolated_func()
except ChildProcessError as e:
    print(f"Function failed: {e}")
```

## Requirements

- Python 3.12+
- UV package manager

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Related Projects

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver