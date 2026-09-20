"""Shared test scaffolding.

Two things were repeated across every test module and are now here: the
default device and the unrolled-linalg switch.  Nothing in this file
changes what is tested -- it only stops ten copies of the same lines
drifting apart.
"""

import mlx.core as mx
import pytest

from anvilgp import linalg

#: float64 arrays cannot be *created* on the GPU stream, so the default is
#: the CPU and the float32 paths opt into the GPU explicitly with
#: ``with mx.stream(mx.gpu):``.  Several tests depend on this: MLX's CPU
#: float32 backend carries extra precision, so a float32 accuracy test that
#: forgets to ask for the GPU can pass while the production path fails.
mx.set_default_device(mx.cpu)


@pytest.fixture
def unrolled():
    """Force the hand-rolled small-matrix path.

    ``linalg`` dispatches to ``mx.matmul`` above a work threshold; the
    unrolled path is the one the Metal kernel mirrors, so the tests that
    pin the kernel's arithmetic pin *that* one.
    """
    linalg.FORCE_UNROLLED = True
    try:
        yield
    finally:
        linalg.FORCE_UNROLLED = False
