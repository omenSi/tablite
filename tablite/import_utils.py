import numpy as np
import math
import psutil
from pathlib import Path
import pyexcel
import sys
import warnings
import logging

from mplite import TaskManager, Task

from datatypes import DataTypes
from config import Config
from file_reader_utils import TextEscape, get_encoding, get_delimiter
from utils import type_check, unique_name, sub_cls_check
from base import Table, Page, Column

from tqdm import tqdm as _tqdm

logging.getLogger("lml").propagate = False
logging.getLogger("pyexcel_io").propagate = False
logging.getLogger("pyexcel").propagate = False


def from_pandas(T, df):
    """
    Creates Table using pd.to_dict('list')

    similar to:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
    >>> df
        a  b
        0  1  4
        1  2  5
        2  3  6
    >>> df.to_dict('list')
    {'a': [1, 2, 3], 'b': [4, 5, 6]}

    >>> t = Table.from_dict(df.to_dict('list))
    >>> t.show()
        +===+===+===+
        | # | a | b |
        |row|int|int|
        +---+---+---+
        | 0 |  1|  4|
        | 1 |  2|  5|
        | 2 |  3|  6|
        +===+===+===+
    """
    sub_cls_check(T, Table)
    return T(columns=df.to_dict("list"))  # noqa


def from_hdf5(T, path):
    """
    imports an exported hdf5 table.
    """
    sub_cls_check(T, Table)

    import h5py

    type_check(path, Path)
    t = T()
    with h5py.File(path, "r") as h5:
        for col_name in h5.keys():
            dset = h5[col_name]
            t[col_name] = dset[:]
    return t


def from_json(T, jsn):
    """
    Imports tables exported using .to_json
    """
    sub_cls_check(T, Table)
    import json

    type_check(jsn, bytes)
    return T(columns=json.loads(jsn))


def excel_reader(T, path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from excel

    **kwargs are excess arguments that are ignored.
    """
    sub_cls_check(T, Table)

    book = pyexcel.get_book(file_name=str(path))

    if sheet is None:  # help the user.
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in book]}")
    elif sheet not in {ws.name for ws in book}:
        raise ValueError(f"sheet not found: {sheet}")

    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >=0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as integer > 0")

    # import a sheets
    for ws in book:
        if ws.name != sheet:
            continue
        else:
            break
    if ws.name != sheet:
        raise ValueError(f'sheet "{sheet}" not found:\n\tSheets: {[str(ws.name) for ws in book]}')

    if columns is None:
        if first_row_has_headers:
            columns = [i[0] for i in ws.columns()]
        else:
            columns = [str(i) for i in range(len(ws.columns()))]

    used_columns_names = set()
    t = T()
    for idx, col in enumerate(ws.columns()):
        if first_row_has_headers:
            header, start_row_pos = str(col[0]), max(1, start)
        else:
            header, start_row_pos = str(idx), max(0, start)

        if header not in columns:
            continue

        unique_column_name = unique_name(str(header), used_columns_names)
        used_columns_names.add(unique_column_name)

        t[unique_column_name] = [v for v in col[start_row_pos : start_row_pos + limit]]
    return t


def ods_reader(T, path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from .ODS
    """
    sub_cls_check(T, Table)
    sheets = pyexcel.get_book_dict(file_name=str(path))

    if sheet is None or sheet not in sheets:
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in sheets]}")

    data = sheets[sheet]
    for _ in range(len(data)):  # remove empty lines at the end of the data.
        if "" == "".join(str(i) for i in data[-1]):
            data = data[:-1]
        else:
            break

    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >=0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as integer > 0")

    t = T()

    used_columns_names = set()
    for ix, value in enumerate(data[0]):
        if first_row_has_headers:
            header, start_row_pos = str(value), 1
        else:
            header, start_row_pos = f"_{ix + 1}", 0

        if columns is not None:
            if header not in columns:
                continue

        unique_column_name = unique_name(str(header), used_columns_names)
        used_columns_names.add(unique_column_name)

        t[unique_column_name] = [row[ix] for row in data[start_row_pos : start_row_pos + limit] if len(row) > ix]
    return t


class TRconfig(object):
    def __init__(
        self,
        source,
        destination,
        column_index,
        start,
        end,
        guess_datatypes,
        delimiter,
        text_qualifier,
        text_escape_openings,
        text_escape_closures,
        strip_leading_and_tailing_whitespace,
        encoding,
    ) -> None:
        self.source = source
        self.destination = destination
        self.column_index = column_index
        self.start = start
        self.end = end
        self.guess_datatypes = guess_datatypes
        self.delimiter = delimiter
        self.text_qualifier = text_qualifier
        self.text_escape_openings = text_escape_openings
        self.text_escape_closures = text_escape_closures
        self.strip_leading_and_tailing_whitespace = strip_leading_and_tailing_whitespace
        self.encoding = encoding
        type_check(column_index, int),
        type_check(start, int),
        type_check(end, int),
        type_check(delimiter, str),
        type_check(text_qualifier, (str, type(None))),
        type_check(text_escape_openings, str),
        type_check(text_escape_closures, str),
        type_check(encoding, str),
        type_check(strip_leading_and_tailing_whitespace, bool),

    def copy(self):
        return TRconfig(**self.dict())

    def dict(self):
        return {k: v for k, v in self.__dict__() if not (k.startswith("_") or callable(v))}


def text_reader_task(
    source,
    destination,
    column_index,
    start,
    end,
    guess_datatypes,
    delimiter,
    text_qualifier,
    text_escape_openings,
    text_escape_closures,
    strip_leading_and_tailing_whitespace,
    encoding,
):
    """PARALLEL TASK FUNCTION
    reads columnsname + path[start:limit] into hdf5.

    source: csv or txt file
    destination: filename for page.
    column_index: int: column index
    start: int: start of page.
    end: int: end of page.
    guess_datatypes: bool: if True datatypes will be inferred by datatypes.Datatypes.guess
    delimiter: ',' ';' or '|'
    text_qualifier: str: commonly \"
    text_escape_openings: str: default: "({[
    text_escape_closures: str: default: ]})"
    strip_leading_and_tailing_whitespace: bool

    encoding: chardet encoding ('utf-8, 'ascii', ..., 'ISO-22022-CN')
    """
    if isinstance(source, str):
        source = Path(source)
    type_check(source, Path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if isinstance(destination, str):
        destination = Path(destination)
    type_check(destination, Path)

    type_check(column_index, int)

    # declare CSV dialect.
    text_escape = TextEscape(
        text_escape_openings,
        text_escape_closures,
        text_qualifier=text_qualifier,
        delimiter=delimiter,
        strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
    )
    values = []
    with source.open("r", encoding=encoding) as fi:  # --READ
        for ix, line in enumerate(fi):
            if ix < start:
                continue
            if ix >= end:
                break
            L = text_escape(line)
            try:
                values.append(L[column_index])
            except IndexError:
                values.append(None)

    array = np.array(DataTypes.guess(values)) if guess_datatypes else np.array(values)
    np.save(destination, array, allow_pickle=True, fix_imports=False)


def text_reader(
    T,
    path,
    columns,
    first_row_has_headers,
    encoding,
    start,
    limit,
    newline,
    guess_datatypes,
    text_qualifier,
    strip_leading_and_tailing_whitespace,
    delimiter,
    text_escape_openings,
    text_escape_closures,
    tqdm=_tqdm,
    **kwargs,
):
    """
    reads any text file

    excess kwargs are ignored.
    """

    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.stat().st_size == 0:
        return T()  # NO DATA: EMPTY TABLE.

    if encoding is None:
        encoding = get_encoding(path)

    if delimiter is None:
        try:
            delimiter = get_delimiter(path, encoding)
        except ValueError:
            return T()  # NO DELIMITER: EMPTY TABLE.

    read_stage, process_stage, dump_stage, consolidation_stage = 20, 50, 20, 10

    pbar_fname = path.name

    if len(pbar_fname) > 20:
        pbar_fname = pbar_fname[0:10] + "..." + pbar_fname[-7:]

    assert sum([read_stage, process_stage, dump_stage, consolidation_stage]) == 100, "Must add to to a 100"

    file_length = path.stat().st_size  # 9,998,765,432 = 10Gb

    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >= 0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as an integer > 0")

    # fmt:off
    with tqdm(total=100, desc=f"importing: reading '{pbar_fname}' bytes", unit="%",
              bar_format="{desc}: {percentage:3.2f}%|{bar}| [{elapsed}<{remaining}]") as pbar:
        # fmt:on
        with path.open("r", encoding=encoding, errors="ignore") as fi:
            # task: find chunk ...
            # Here is the problem in a nutshell:
            # --------------------------------------------------------
            # text = "this is my \n text".encode('utf-16')
            # >>> text
            # b'\xff\xfet\x00h\x00i\x00s\x00 \x00i\x00s\x00 \x00m\x00y\x00 \x00\n\x00 \x00t\x00e\x00x\x00t\x00'
            # >>> newline = "\n".encode('utf-16')
            # >>> newline in text
            # False
            # >>> newline.decode('utf-16') in text.decode('utf-16')
            # True
            # --------------------------------------------------------
            # This means we can't read the encoded stream to check if in contains a particular character.
            # We will need to decode it.
            # furthermore fi.tell() will not tell us which character we a looking at.
            # so the only option left is to read the file and split it in workable chunks.
            try:
                newlines = 0
                for block in fi:
                    newlines = newlines + 1

                    pbar.update((len(block) / file_length) * read_stage)

                pbar.desc = f"importing: processing '{pbar_fname}'"
                pbar.update(read_stage - pbar.n)

                fi.seek(0)
            except Exception as e:
                raise ValueError(f"file could not be read with encoding={encoding}\n{str(e)}")
            if newlines < 1:
                raise ValueError(f"Using {newline} to split file, revealed {newlines} lines in the file.")

            if newlines <= start + (1 if first_row_has_headers else 0):  # Then start > end: Return EMPTY TABLE.
                return Table(columns={n : [] for n in columns})

        line_reader = TextEscape(
            openings=text_escape_openings,
            closures=text_escape_closures,
            text_qualifier=text_qualifier,
            delimiter=delimiter,
            strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
        )

        with path.open("r", encoding=encoding) as fi:
            for line in fi:
                if line == "":  # skip any empty line.
                    continue
                else:
                    line = line.rstrip(newline)
                    fields = line_reader(line)
                    break
            if not fields:
                warnings.warn("file was empty: {path}")
                return T()  # returning an empty table as there was no data.

            if len(set(fields)) != len(fields):  # then there's duplicate names.
                new_fields = []
                for name in fields:
                    new_fields.append(unique_name(name, new_fields))
                # header_line = delimiter.join(new_fields) + newline
                fields = new_fields

        cpu_count = max(psutil.cpu_count(logical=True), 1)  # there's always at least one core!
        tasks = math.ceil(newlines / Config.PAGE_SIZE) * len(columns)

        task_config = TRconfig(
            source=str(path),
            destination=None,
            column_index=0,
            start=1,
            end=Config.PAGE_SIZE,
            guess_datatypes=guess_datatypes,
            delimiter=delimiter,
            text_qualifier=text_qualifier,
            text_escape_openings=text_escape_openings,
            text_escape_closures=text_escape_closures,
            strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
            encoding=encoding,
        )
        tasks, configs = [], {}
        for ix, field_name in enumerate(fields):
            subconfig = [field_name]
            configs.append(subconfig)

            start = end = 1 if first_row_has_headers else 0
            for end in range(start + Config.PAGE_SIZE, newlines + 1, Config.PAGE_SIZE):

                cfg = task_config.copy()
                cfg.start = start
                cfg.end = min(end, newlines)
                cfg.destination = next(Page.ids)
                cfg.column_index = ix
                tasks.append(Task(f=text_reader_task, **cfg.dict()))
                subconfig.append(cfg)

                start = end

        pbar.desc = f"importing: saving '{pbar_fname}' to disk"
        pbar.update((read_stage + process_stage) - pbar.n)

        len_tasks = len(tasks)
        dump_size = dump_stage / len_tasks

        class PatchTqdm:  # we need to re-use the tqdm pbar, this will patch
            # the tqdm to update existing pbar instead of creating a new one
            def update(self, n=1):
                pbar.update(n * dump_size)

        if cpu_count < 2 or Config.MULTIPROCESSING_ENABLED is False:
            for task in tasks:
                task.execute()
                pbar.update(dump_size)
        else:
            with TaskManager(cpu_count - 1) as tm:
                errors = tm.execute(tasks, pbar=PatchTqdm())  # I expects a list of None's if everything is ok.
                if any(errors):
                    raise Exception("\n".join(e for e in errors if e))

        pbar.desc = f"importing: consolidating '{pbar_fname}'"
        pbar.update((read_stage + process_stage + dump_stage) - pbar.n)

        consolidation_size = consolidation_stage / len_tasks

        # consolidate the task results
        t = T()
        for subconfig in configs:
            name, pages = subconfig[0], subconfig[1:]
            if name in t.columns:
                name = unique_name(name, set(t.columns))
            col = Column(t.path)
            col.pages.extend(pages[:])
            t.columns[name] = col

            pbar.update(consolidation_size)

        pbar.update(100 - pbar.n)
        return t


file_readers = {  # dict of file formats and functions used during Table.import_file
    "fods": excel_reader,
    "json": excel_reader,
    "html": excel_reader,
    "simple": excel_reader,
    "rst": excel_reader,
    "mediawiki": excel_reader,
    "xlsx": excel_reader,
    "xls": excel_reader,
    "xlsm": excel_reader,
    "csv": text_reader,
    "tsv": text_reader,
    "txt": text_reader,
    "ods": ods_reader,
}

valid_readers = ",".join(list(file_readers.keys()))
