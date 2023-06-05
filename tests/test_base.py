from tablite.base import Table, Column, Page
from tablite.config import Config
import numpy as np
from pathlib import Path
import math
import os

import logging

log = logging.getLogger("base")
log.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(level=logging.DEBUG)
formatter = logging.Formatter("%(levelname)s : %(message)s")
console.setFormatter(formatter)
log.addHandler(console)


def test_basics():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = Table(columns=data)
    assert t == data
    assert t.items() == data.items()

    a = t["A"]  # selects column 'a'
    assert isinstance(a, Column)
    b = t[3]  # selects row 3 as a tuple.
    assert isinstance(b, tuple)
    c = t[:10]  # selects first 10 rows from all columns
    assert isinstance(c, Table)
    assert len(c) == 10
    assert c.items() == {k: v[:10] for k, v in data.items()}.items()
    d = t["A", "B", slice(3, 20, 2)]  # selects a slice from columns 'a' and 'b'
    assert isinstance(d, Table)
    assert len(d) == 9
    assert d.items() == {k: v[3:20:2] for k, v in data.items() if k in ("A", "B")}.items()

    e = t["B", "A"]  # selects column 'b' and 'c' and 'a' twice for a slice.
    assert list(e.columns) == ["B", "A"], "order not maintained."

    x = t["A"]
    assert isinstance(x, Column)
    assert x == list(A)
    assert x == np.array(A)
    assert x == tuple(A)


def test_empty_table():
    t2 = Table()
    assert isinstance(t2, Table)
    t2["A"] = []
    assert len(t2) == 0
    c = t2["A"]
    assert isinstance(c, Column)
    assert len(c) == 0


def test_page_size():
    original_value = Config.PAGE_SIZE

    Config.PAGE_SIZE = 10
    assert Config.PAGE_SIZE == 10
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    # during __setitem__ Column.paginate will use config.
    t = Table(columns=data)
    assert t == data
    assert t.items() == data.items()

    x = t["A"]
    assert isinstance(x, Column)
    assert len(x.pages) == math.ceil(len(A) / Config.PAGE_SIZE)
    Config.PAGE_SIZE = 7
    t2 = Table(columns=data)
    assert t == t2
    assert not t != t2
    x2 = t2["A"]
    assert isinstance(x2, Column)
    assert len(x2.pages) == math.ceil(len(A) / Config.PAGE_SIZE)

    Config.reset()
    assert Config.PAGE_SIZE == original_value


def test_cleaup():
    A = list(range(1, 10_000_000))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = Table(columns=data)
    assert isinstance(t, Table)
    _folder = t._pid_dir
    _t_path = t.path

    del t
    import gc

    while gc.collect() > 0:
        pass

    assert _folder.exists()  # should be there until sigint.
    assert not _t_path.exists()  # should have been deleted


def save_and_load():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = Table(columns=data)
    assert isinstance(t, Table)
    my_folder = Path(Config.workdir) / "data"
    my_folder.mkdir(exist_ok=True)
    my_file = my_folder / "my_first_file.tpz"
    t.save(my_file)
    assert my_file.exists()
    assert os.path.getsize(my_file) > 0

    del t
    import gc

    while gc.collect() > 0:
        pass

    assert my_file.exists()
    t2 = Table.load(my_file)

    t3 = Table(columns=data)
    assert t2 == t3
    assert t2.path.parent == t3.path.parent, "t2 must be loaded into same PID dir as t3"

    del t2
    while gc.collect() > 0:
        pass

    assert my_file.exists(), "deleting t2 MUST not affect the file saved by the user."
    t4 = Table.load(my_file)
    assert t4 == t3

    os.remove(my_file)
    os.rmdir(my_folder)


def test_copy():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t1 = Table(columns=data)
    assert isinstance(t1, Table)

    t2 = t1.copy()
    for name, t2column in t2.columns.items():
        t1column = t1.columns[name]
        assert id(t2column) != id(t1column)
        for p2, p1 in zip(t2column.pages, t1column.pages):
            assert id(p2) == id(p1), "columns can point to the same pages."


def test_speed():
    log.setLevel(logging.INFO)
    A = list(range(1, 10_000_000))
    B = [i * 10 for i in A]
    data = {"A": A, "B": B}
    t = Table(columns=data)
    import time
    import random

    loops = 100
    random.seed(42)
    start = time.time()
    for i in range(loops):
        a, b = random.randint(1, len(A)), random.randint(1, len(A))
        a, b = min(a, b), max(a, b)
        block = t["A"][a:b]  # pure numpy.
        assert len(block) == b - a
    end = time.time()
    print(f"numpy array: {end-start}")

    random.seed(42)
    start = time.time()
    for i in range(loops):
        a, b = random.randint(1, len(A)), random.randint(1, len(A))
        a, b = min(a, b), max(a, b)
        block = t["A"][a:b].tolist()  # python.
        assert len(block) == b - a
    end = time.time()
    print(f"python list: {end-start}")


def test_immutability_of_pages():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)
    change = t["A"][7:3:-1]
    t["A"][3:7] = change


def test_slice_functions():
    Config.PAGE_SIZE = 3
    t = Table(columns={"A": np.array([1, 2, 3, 4])})
    L = t["A"]
    assert list(L[:]) == [1, 2, 3, 4]  # slice(None,None,None)
    assert list(L[:0]) == []  # slice(None,0,None)
    assert list(L[0:]) == [1, 2, 3, 4]  # slice(0,None,None)

    assert list(L[:2]) == [1, 2]  # slice(None,2,None)

    assert list(L[-1:]) == [4]  # slice(-1,None,None)
    assert list(L[-1::1]) == [4]
    assert list(L[-1:None:1]) == [4]

    # assert list(L[-1:4:1]) == [4]  # slice(-1,4,1)
    # assert list(L[-1:0:-1]) == [4, 3, 2]  # slice(-1,0,-1) -->  slice(4,0,-1) --> for i in range(4,0,-1)
    # assert list(L[-1:0:1]) == []  # slice(-1,0,1) --> slice(4,0,-1) --> for i in range(4,0,1)
    # assert list(L[-3:-1:1]) == [2, 3]

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t = Table(columns={"A": np.array(data)})
    L = t["A"]
    assert list(L[-1:0:-1]) == data[-1:0:-1]
    assert list(L[-1:0:1]) == data[-1:0:1]

    data = list(range(100))
    t2 = Table(columns={"A": np.array(data)})
    L = t2["A"]
    assert list(L[51:40:-1]) == data[51:40:-1]
    Config.reset()


def test_various():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)
    t *= 2
    assert len(t) == len(A) * 2
    x = t["A"]
    assert len(x) == len(A) * 2

    t2 = t.copy()
    t2 += t
    assert len(t2) == len(t) * 2
    assert len(t2["A"]) == len(t["A"]) * 2
    assert len(t2["B"]) == len(t["B"]) * 2
    assert len(t2["C"]) == len(t["C"]) * 2

    assert t != t2

    orphaned_column = t["A"].copy()
    orphaned_column_2 = orphaned_column * 2
    t3 = Table()
    t3["A"] = orphaned_column_2

    orphaned_column.replace(mapping={2: 20, 3: 30, 4: 40})
    z = set(orphaned_column[:].tolist())
    assert {2, 3, 4}.isdisjoint(z)
    assert {20, 30, 40}.issubset(z)

    t4 = t + t2
    assert len(t4) == len(t) + len(t2)


def test_types():
    # SINGLE TYPES.
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    c = t["A"]
    assert c.types() == {int: len(A)}

    assert t.types() == {"A": {int: len(A)}, "B": {int: len(A)}, "C": {int: len(A)}}

    # MIXED DATATYPES
    A = list(range(5))
    B = list("abcde")
    data = {"A": A, "B": B}
    t = Table(columns=data)
    typ1 = t.types()
    expected = {"A": {int: 5}, "B": {str: 5}}
    assert typ1 == expected
    more = {"A": B, "B": A}
    t += Table(columns=more)
    typ2 = t.types()
    expected2 = {"A": {int: 5, str: 5}, "B": {str: 5, int: 5}}
    assert typ2 == expected2


def test_table_row_functions():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    t2 = Table()
    t2.add_column("A")
    t2.add_columns("B", "C")
    t2.add_rows(**{"A": [i + max(A) for i in A], "B": [i + max(A) for i in A], "C": [i + max(C) for i in C]})
    t3 = t2.stack(t)
    assert len(t3) == len(t2) + len(t)

    t4 = Table(columns={"B": [-1, -2], "D": [0, 1]})
    t5 = t2.stack(t4)
    assert len(t5) == len(t2) + len(t4)
    assert list(t5.columns) == ["A", "B", "C", "D"]


def test_remove_all():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    c = t["A"]
    c.remove_all(3)
    A.remove(3)
    assert list(c) == A
    c.remove_all(4, 5, 6)
    A = [i for i in A if i not in {4, 5, 6}]
    assert list(c) == A


def test_replace():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    c = t["A"]
    c.replace({3: 30, 4: 40})
    assert list(c) == [i if i not in {3, 4} else i * 10 for i in A]


def test_display_options():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    d = t.display_dict()
    assert d == {
        "#": [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"],
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "C": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }

    txt = t.to_ascii()
    expected = """\
+==+==+===+====+
|# |A | B | C  |
+--+--+---+----+
| 0| 1| 10| 100|
| 1| 2| 20| 200|
| 2| 3| 30| 300|
| 3| 4| 40| 400|
| 4| 5| 50| 500|
| 5| 6| 60| 600|
| 6| 7| 70| 700|
| 7| 8| 80| 800|
| 8| 9| 90| 900|
| 9|10|100|1000|
+==+==+===+====+"""
    assert txt == expected
    txt2 = t.to_ascii(dtype=True)
    expected2 = """\
+===+===+===+====+
| # | A | B | C  |
|row|int|int|int |
+---+---+---+----+
| 0 |  1| 10| 100|
| 1 |  2| 20| 200|
| 2 |  3| 30| 300|
| 3 |  4| 40| 400|
| 4 |  5| 50| 500|
| 5 |  6| 60| 600|
| 6 |  7| 70| 700|
| 7 |  8| 80| 800|
| 8 |  9| 90| 900|
| 9 | 10|100|1000|
+===+===+===+====+"""
    assert txt2 == expected2

    html = t._repr_html_()

    expected3 = """<div><table border=1>\
<tr><th>#</th><th>A</th><th>B</th><th>C</th></tr>\
<tr><th> 0</th><th>1</th><th>10</th><th>100</th></tr>\
<tr><th> 1</th><th>2</th><th>20</th><th>200</th></tr>\
<tr><th> 2</th><th>3</th><th>30</th><th>300</th></tr>\
<tr><th> 3</th><th>4</th><th>40</th><th>400</th></tr>\
<tr><th> 4</th><th>5</th><th>50</th><th>500</th></tr>\
<tr><th> 5</th><th>6</th><th>60</th><th>600</th></tr>\
<tr><th> 6</th><th>7</th><th>70</th><th>700</th></tr>\
<tr><th> 7</th><th>8</th><th>80</th><th>800</th></tr>\
<tr><th> 8</th><th>9</th><th>90</th><th>900</th></tr>\
<tr><th> 9</th><th>10</th><th>100</th><th>1000</th></tr>\
</table></div>"""
    assert html == expected3

    A = list(range(1, 51))
    B = [i * 10 for i in A]
    C = [i * 1000 for i in B]
    data = {"A": A, "B": B, "C": C}
    t2 = Table(columns=data)
    d2 = t2.display_dict()
    assert d2 == {
        "#": [" 0", " 1", " 2", " 3", " 4", " 5", " 6", "...", "43", "44", "45", "46", "47", "48", "49"],
        "A": [1, 2, 3, 4, 5, 6, 7, "...", 44, 45, 46, 47, 48, 49, 50],
        "B": [10, 20, 30, 40, 50, 60, 70, "...", 440, 450, 460, 470, 480, 490, 500],
        "C": [
            10000,
            20000,
            30000,
            40000,
            50000,
            60000,
            70000,
            "...",
            440000,
            450000,
            460000,
            470000,
            480000,
            490000,
            500000,
        ],
    }


def test_index():
    A = [1, 1, 2, 2, 3, 3, 4]
    B = list("asdfghjkl"[: len(A)])
    data = {"A": A, "B": B}
    t = Table(columns=data)
    t.index("A")
    t.index("A", "B")


def test_to_dict():
    # to_dict and as_json_serializable
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)
    assert t.to_dict() == data


def test_unique():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert np.all(t["A"].unique() == np.array([1, 2, 3, 4]))


def test_histogram():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert t["A"].histogram() == {1: 3, 2: 3, 3: 2, 4: 1}


def test_count():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert t["A"].count(2) == 3


def test_contains():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert 3 in t["A"]
    assert 7 not in t["A"]