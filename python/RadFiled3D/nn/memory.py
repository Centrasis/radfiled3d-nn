"""``RadFiled3D.nn.memory`` — buffers inference reads and writes, mirroring C++ ``::nn::memory``.

A buffer crosses **by pointer and is never copied**: a torch CUDA tensor stays on the GPU for the
whole run. Anything contiguous and float32 works — a NumPy array, a torch tensor, a memoryview —
recognised through ``__array_interface__``, ``__cuda_array_interface__`` or the buffer protocol.
Non-contiguous input raises rather than being quietly copied, because a copy is not the buffer the
caller asked to have written.
"""

from ._rfnn.memory import Memory

__all__ = ["Memory"]
