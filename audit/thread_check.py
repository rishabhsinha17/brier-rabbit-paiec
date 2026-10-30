"""Does importing numpy (OpenBLAS build) before model.py's thread pinning slow the shipped model? One B31 checkpoint
slice (refit + 60 calls) with numpy imported first, BLAS threads unpinned vs pinned by the environment.
usage: [OPENBLAS_NUM_THREADS=1] python3.11 thread_check.py"""
import os, sys, time
import numpy  # imported BEFORE the shipped model, so model.py's os.environ.setdefault cannot pin BLAS threads
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
mod, d = common.shipped("v6_combo_w20")
labeled, inputs = common.draw(common.prior_names(d), budget=31, seed=0)
t0 = time.perf_counter()
for inp in inputs[:60]:
    mod.predict(inp, labeled)
print(f"numpy {numpy.__version__}, OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}: "
      f"refit + 59 calls {time.perf_counter() - t0:.2f}s", flush=True)
