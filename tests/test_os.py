import time
import math
import tablite.h5py as h5py
import pathlib
import tempfile

from tablite.config import TEMPDIR
from tablite.core import _text_reader_task_size


def test01():
    assert TEMPDIR.exists()


def test2():
    t = tempfile.gettempdir()
    path = pathlib.Path(t) / "my.h5"
    if path.exists():
        path.unlink()

    key = "/table/1"
    with h5py.File(path, "a") as h5:
        dset = h5.create_dataset(name=key, dtype=h5py.Empty("f"))
        dset.attrs["now"] = time.time()

    with h5py.File(path, "w") as h5:
        assert list(h5.keys()) == []

    with h5py.File(path, "r+") as h5:
        assert key not in h5.keys()
        dset = h5.create_dataset(name=key, dtype=h5py.Empty("f"))
        dset.attrs["now"] = time.time()

    path.unlink()

def test_one_core():
    newlines = 10_000
    filesize = 1_000_000 # ~1MB
    cpu_count = 1
    free_virtual_memory = 8_000_000_000 # ~8GB
    
    lines_per_task, _ = _text_reader_task_size(newlines, filesize, cpu_count, free_virtual_memory)
    assert (newlines / lines_per_task) == 1, "Should fit in one task!"

    filesize = 16_000_000_000 # ~8GB
    lines_per_task, _ = _text_reader_task_size(newlines, filesize, cpu_count, free_virtual_memory, working_overhead=0, memory_usage_ceiling=1)
    assert (newlines / lines_per_task) == 2, "Should fit in two tasks!"

    filesize = 8_000_000_000 # ~8GB
    lines_per_task, _ = _text_reader_task_size(newlines, filesize, cpu_count, free_virtual_memory, working_overhead=0)
    assert (newlines / lines_per_task) == 2, "Should fit in two tasks!"

def test_multi_core():
    newlines = 10_000
    filesize = 8_000_000_000 # ~8GB
    cpu_count = 2
    free_virtual_memory = 8_000_000_000 # ~8GB
    
    lines_per_task, cpu_count = _text_reader_task_size(newlines, filesize, cpu_count, free_virtual_memory, working_overhead=0, memory_usage_ceiling=1,python_mem_w_imports=0)

    assert (newlines / lines_per_task) == 2, "Should fit in two tasks!"
    assert cpu_count == 2, "Should be two CPUs!"

    cpu_count = 3
    lines_per_task, cpu_count = _text_reader_task_size(newlines, filesize, cpu_count, free_virtual_memory, working_overhead=0, memory_usage_ceiling=1,python_mem_w_imports=0)

    assert math.ceil(newlines / lines_per_task) == 3, "Should fit in three tasks!"
    assert cpu_count == 3, "Should be three CPUs!"