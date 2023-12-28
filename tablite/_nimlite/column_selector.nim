import std/[tables, sugar, sets, sequtils, strutils]
import nimpy as nimpy
import utils
import std/options as opt
import pymodules as pymodules
from numpy import getColumnLen, getColumnTypes
from unpickling import PageTypes

type DesiredColumnInfo = object
    original_name: string
    `type`: PageTypes
    allow_empty: bool

proc toPageType(name: string): PageTypes =
    case name.toLower():
    of "int": return PageTypes.DT_INT
    of "float": return PageTypes.DT_FLOAT
    of "bool": return PageTypes.DT_BOOL
    of "str": return PageTypes.DT_STRING
    of "date": return PageTypes.DT_DATE
    of "time": return PageTypes.DT_TIME
    of "datetime": return PageTypes.DT_DATETIME
    else: raise newException(FieldDefect, "unsupported page type: '" & name & "'")

proc columnSelect(table: nimpy.PyObject, cols: nimpy.PyObject, tqdm: nimpy.PyObject, dir_pid: string): (nimpy.PyObject, nimpy.PyObject) =
    var desired_column_map = initTable[string, DesiredColumnInfo]()
    var collisions = initTable[string, int]()

     ######################################################
    # 1. Figure out what user needs (types, column names)
    ######################################################
    for c in cols:                              # now lets iterate over all given columns
        # this is our old name
        let name_inp = c["column"].to(string)
        var rename = c.get("rename", builtins().None)

        if rename.isNone() or not builtins().isinstance(rename, builtins().str).to(bool) or builtins().len(rename).to(int) == 0:
            rename = builtins().None
        else:
            let name_out_stripped = rename.strip()
            if builtins().len(rename).to(int) > 0 and builtins().len(name_out_stripped).to(int) == 0:
                raise newException(ValueError, "Validating 'column_select' failed, '" & name_inp & "' cannot be whitespace.")

            rename = name_out_stripped

        var name_out = if rename.isNone(): name_inp else: rename.to(string)

        if name_out in collisions:              # check if the new name has any collision, add suffix if necessary
            collisions[name_out] = collisions[name_out] + 1
            name_out = name_out & "_" & $(collisions[name_out] - 1)
        else:
            collisions[name_out] = 1

        let desired_type = c.get("type", builtins().None)

        desired_column_map[name_out] = DesiredColumnInfo(   # collect the information about the column, fill in any defaults
            original_name: name_inp,
            `type`: if desired_type.isNone(): PageTypes.DT_NONE else: toPageType(desired_type.to(string)),
            allow_empty: c.get("allow_empty", builtins().False).to(bool)
        )

    ######################################################
    # 2. Converting types to user specified
    ######################################################
    let columns = collect(initTable()):
        for pyColName in table.columns:
            let colName = pyColName.to(string)
            let pyColPages = table[colName].pages
            let pages = collect: 
                for pyPage in pyColPages: 
                    builtins().str(pyPage.path.absolute()).to(string)

            {colName: pages}

    let column_names = collect: (for k in columns.keys: k)
    var layout_set = newSeq[(int, int)]()

    for pages in columns.values():
        let pgCount = pages.len
        let elCount = getColumnLen(pages)
        let layout = (elCount, pgCount)

        if layout in layout_set:
            continue

        if layout_set.len != 0:
            raise newException(RangeDefect, "Data layout mismatch, pages must be consistent")

        layout_set.add(layout)
        
    let page_count = layout_set[0][0]

      # Registry of data
    var passed_column_data = initTable[string, seq[string]]()
    var failed_column_data = initTable[string, seq[string]]()

    var cols = initTable[string, seq[string]]()

    var res_cols_pass = collect: (for _ in 0..(page_count-1): initTable[string, seq[string]])
    var res_cols_fail = collect: (for _ in 0..(page_count-1): initTable[string, seq[string]])

    var is_correct_type = initTable[string, bool]()

    for (desired_name_non_unique, desired_columns) in desired_column_map.pairs():
        let keys = passed_column_data.toSeqKeys()
        let desired_name = unique_name(desired_name_non_unique, keys)
        let this_col = columns[desired_columns.original_name]
        
        cols[desired_name] = this_col
        
        passed_column_data[desired_name] = @[]

        var col_dtypes = this_col.getColumnTypes().toSeqKeys()
        var needs_to_iterate = false

        if PageTypes.DT_NONE in col_dtypes:
            if not desired_columns.allow_empty:
                needs_to_iterate = true
            else:
                col_dtypes.delete(col_dtypes.find(PageTypes.DT_NONE))

        if not needs_to_iterate and col_dtypes.len > 0:
            if col_dtypes.len > 1:
                needs_to_iterate = true
            else:
                let active_type = col_dtypes[0]

                if active_type != desired_columns.`type`:
                    needs_to_iterate = true

        is_correct_type[desired_name] = not needs_to_iterate

        # for i in range(page_count):
        #     res_cols_pass[i][desired_name] = (dir_pid, SimplePage.next_id(dir_pid))

    # for desired_name in table.columns:
    #     failed_column_data[desired_name] = []

    #     for i in range(page_count):
    #         res_cols_fail[i][desired_name] = (dir_pid, SimplePage.next_id(dir_pid))

    # reject_reason_name = unique_name("reject_reason", table.columns.keys())
    
    implement("columnSelect")

when isMainModule and appType != "lib":
    proc newColumnSelectorInfo(column: string, `type`: string, allow_empty: bool, rename: opt.Option[string]): nimpy.PyObject =
        let pyDict = builtins().dict()

        pyDict["column"] = column
        pyDict["type"] = `type`
        pyDict["allow_empty"] = allow_empty

        if rename.isNone():
            pyDict["rename"] = nimpy.newPyNone()
        else:
            pyDict["rename"] = rename.get()

        return pyDict

    discard pyImport("sys").path.extend(@[
        "/home/ratchet/envs/callisto/lib/python3.10/site-packages",
        "/home/ratchet/Documents/dematic/tablite",
        "/home/ratchet/Documents/dematic/mplite"
    ])

    let columns = pymodules.builtins().dict({"A ": @[0, 1, 2]}.toTable)
    let table = pymodules.tablite().Table(columns=columns)

    let select_cols = builtins().list(@[
        newColumnSelectorInfo("A ", "int", false, opt.none[string]())
    ])

    let (select_pass, select_fail) = table.columnSelect(
        select_cols,
        nimpy.pyImport("tqdm").tqdm,
        dir_pid="/media/ratchet/hdd/tablite/nim"
    )

    echo select_pass.show()
