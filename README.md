# Machine Learning Helper Package

**Warning: This package is currently underdeveloped and does not guarantee any consistency.**

## Package Introduction

## Package Contents

### utils

**file_helpers**: helper functions for file manipulation, including:
- Partially integrated & improved `os` and `pathlib` functions.
- Loading & saving in `json` and `pickle`.

**view_helpers**: helper functions for console debugging output, including:
- Length-restricting view of objects supporting multiple formats.
- Output stream and color setting.
- `TQDM` wrapping.

**Serializable**: the essence of `mlhp` classes: all classes should be designed to be serializable.

**log_helpers**: a custom logger that does not rely on `logging` handle (and thus does not conflict with DDP training, for details about this issue, please refer to [this post](https://stackoverflow.com/questions/64752343/pytorch-why-logging-fails-in-ddp)). **Notice that we are considering moving this to a new independent `Logger` module.**

**helpers**: general helper functions, including:
- Daily-used system functions (e.g.: `CMD`, `IP`).
- In-program package installation throught `PIP` and `CONDA`.
- Miscellaneous

### Plotter

A `matplotlib.pyplot` handle with `with` wrapping.

### Timer

A `Timer`.

### Tracker

A curve tracker supporting updating, finding improvements, logging and plotting.

(developing... )

### ModelWrapper

A general model wrapper for arbitrary network, training, evaluating and testing pipeline.

(developing... )