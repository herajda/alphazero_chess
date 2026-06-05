"""Public Python shim for the native AlphaZero chess engine extension.

Import torch before loading the pybind/libtorch extension. The Python torch
package performs CUDA/cuDNN runtime initialization that the native module relies
on when running TorchScript models with CuDNN enabled.
"""
import torch as _torch  # noqa: F401

from _chess_engine import *  # noqa: F401,F403
