from tqdm import tqdm
from utils import sub_cls_check, type_check
from table import Table
from datatypes import DataTypes
from pathlib import Path


def to_sql(table):
    """
    generates ANSI-92 compliant SQL.
    """
    sub_cls_check(table, Table)

    prefix = "Table"
    create_table = """CREATE TABLE {}{} ({})"""
    columns = []
    for name, col in table.items():
        dtype = col.types()
        if len(dtype) == 1:
            dtype, _ = dtype.popitem()
            if dtype is int:
                dtype = "INTEGER"
            elif dtype is float:
                dtype = "REAL"
            else:
                dtype = "TEXT"
        else:
            dtype = "TEXT"
        definition = f"{name} {dtype}"
        columns.append(definition)

    create_table = create_table.format(prefix, table.id, ", ".join(columns))

    # return create_table
    row_inserts = []
    for row in table.rows:
        row_inserts.append(str(tuple([i if i is not None else "NULL" for i in row])))
    row_inserts = f"INSERT INTO {prefix}{table.id} VALUES " + ",".join(row_inserts)
    return "begin; {}; {}; commit;".format(create_table, row_inserts)


def to_pandas(table):
    """
    returns pandas.DataFrame
    """
    sub_cls_check(table, Table)
    try:
        return pd.DataFrame(table.to_dict())  # noqa
    except ImportError:
        import pandas as pd  # noqa
    return pd.DataFrame(table.to_dict())  # noqa


def to_hdf5(table, path):
    """
    creates a copy of the table as hdf5
    """
    import h5py

    sub_cls_check(table, Table)
    type_check(path, Path)

    total = ":,".format(len(table.columns) * len(table))  # noqa
    print(f"writing {total} records to {path}")

    with h5py.File(path, "a") as f:
        with tqdm(total=len(table.columns), unit="columns") as pbar:
            n = 0
            for name, col in table.items():
                f.create_dataset(name, data=col[:])  # stored in hdf5 as '/name'
                n += 1
                pbar.update(n)
    print(f"writing {path} to HDF5 done")


def excel_writer(table, path):
    """
    writer for excel files.

    This can create xlsx files beyond Excels.
    If you're using pyexcel to read the data, you'll see the data is there.
    If you're using Excel, Excel will stop loading after 1,048,576 rows.

    See pyexcel for more details:
    http://docs.pyexcel.org/
    """
    import pyexcel

    sub_cls_check(table, Table)
    type_check(path, Path)

    def gen(table):  # local helper
        yield table.columns
        for row in table.rows:
            yield row

    data = list(gen(table))
    if path.suffix in [".xls", ".ods"]:
        data = [
            [
                str(v)
                if (isinstance(v, (int, float)) and abs(v) > 2**32 - 1)
                else DataTypes.to_json(v)
                for v in row
            ]
            for row in data
        ]

    pyexcel.save_as(array=data, dest_file_name=str(path))


def to_json(table, *args, **kwargs):
    sub_cls_check(table, Table)
    return table.to_json(*args, **kwargs)


def text_writer(table, path, tqdm=_tqdm):
    """exports table to csv, tsv or txt dependening on path suffix.
    follows the JSON norm. text escape is ON for all strings.

    Note:
    ----------------------
    If the delimiter is present in a string when the string is exported,
    text-escape is required, as the format otherwise is corrupted.
    When the file is being written, it is unknown whether any string in
    a column contrains the delimiter. As text escaping the few strings
    that may contain the delimiter would lead to an assymmetric format,
    the safer guess is to text escape all strings.
    """
    sub_cls_check(table, Table)
    type_check(path, Path)

    def txt(value):  # helper for text writer
        if value is None:
            return ""  # A column with 1,None,2 must be "1,,2".
        elif isinstance(value, str):
            # if not (value.startswith('"') and value.endswith('"')):
            #     return f'"{value}"'  # this must be escape: "the quick fox, jumped over the comma"
            # else:
            return value  # this would for example be an empty string: ""
        else:
            return str(
                DataTypes.to_json(value)
            )  # this handles datetimes, timedelta, etc.

    delimiters = {".csv": ",", ".tsv": "\t", ".txt": "|"}
    delimiter = delimiters.get(path.suffix)

    with path.open("w", encoding="utf-8") as fo:
        fo.write(delimiter.join(c for c in table.columns) + "\n")
        for row in tqdm(table.rows, total=len(table)):
            fo.write(delimiter.join(txt(c) for c in row) + "\n")


def sql_writer(table, path):
    type_check(table, Table)
    type_check(path, Path)
    with path.open("w", encoding="utf-8") as fo:
        fo.write(to_sql(table))


def json_writer(table, path):
    type_check(table, Table)
    type_check(path, Path)
    with path.open("w") as fo:
        fo.write(to_json(table))


def h5_writer(table, path):
    type_check(table, Table)
    type_check(path, Path)
    to_hdf5(table, path)


def html_writer(table, path):
    type_check(table, Table)
    type_check(path, Path)
    with path.open("w", encoding="utf-8") as fo:
        fo.write(table._repr_html_())


exporters = {  # the commented formats are not yet supported by the pyexcel plugins:
    # 'fods': excel_writer,
    "json": json_writer,
    "html": html_writer,
    # 'simple': excel_writer,
    # 'rst': excel_writer,
    # 'mediawiki': excel_writer,
    "xlsx": excel_writer,
    "xls": excel_writer,
    # 'xlsm': excel_writer,
    "csv": text_writer,
    "tsv": text_writer,
    "txt": text_writer,
    "ods": excel_writer,
    "sql": sql_writer,
    # 'hdf5': h5_writer,
    # 'h5': h5_writer
}

supported_formats = list(exporters.keys())
