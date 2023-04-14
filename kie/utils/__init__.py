import json
import numpy as np
import time

from .io import *
from .data import *


class Timer:
    def __init__(self, msg=None):
        self.t = 0
        self.msg = msg

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a, **k):
        b = time.perf_counter()
        if self.msg is not None:
            print(f"[{self.msg}] Ellapsed: {b - self.t:.9f}")
        else:
            print(f"Ellapsed: {b - self.t:.9f}")

