#[
    Put all python export wrappers here.

    There is an issue with nimpy where it is impossible to have exports in other files no matter if you use import/include syntax.
    Therefore, we must unfortunately put all Python accessible API's in this file.

    Alternative would be split the API's to separate dynamic libraries bout that adds memory overhead because each of the libraries would duplicate the binding code.
]#


template isLib(): bool = isMainModule and appType == "lib"

when isLib:
    import nimpy
    import std/[paths]
    import std/[os, options, tables]

    # -------- COLUMN SELECTOR --------
    import funcs/column_selector as column_selector

    proc collect_column_select_info*(table: PyObject, cols: PyObject, dir_pid: string): (
        Table[string, seq[string]], int, Table[string, bool], PyObject, seq[string], seq[string], seq[column_selector.ColInfo], seq[column_selector.ColInfo], seq[string], string
    ) {.exportpy.} =
        var (columns, page_count, is_correct_type, desired_column_map, passed_column_data, failed_column_data, res_cols_pass, res_cols_fail, column_names, reject_reason_name) = column_selector.collectColumnSelectInfo(table, cols, dir_pid)

        return (columns, page_count, is_correct_type, desired_column_map.toPyObj, passed_column_data, failed_column_data, res_cols_pass, res_cols_fail, column_names, reject_reason_name)

    proc do_slice_convert*(dir_pid: string, page_size: int, columns: Table[string, string], reject_reason_name: string, res_pass: column_selector.ColInfo, res_fail: column_selector.ColInfo, desired_column_map: PyObject, column_names: seq[string], is_correct_type: Table[string, bool]): (seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)]) {.exportpy.} =
        return column_selector.doSliceConvert(Path(dir_pid), page_size, columns, reject_reason_name, res_pass, res_fail, desired_column_map.fromPyObjToDesiredInfos, column_names, is_correct_type)
    # -------- COLUMN SELECTOR --------

    # --------   TEXT READER   --------
    import funcs/text_reader as text_reader

    proc text_reader_task(
        path: string,
        encoding: string,
        dia_delimiter: string,
        dia_quotechar: string,
        dia_escapechar: string,
        dia_doublequote: bool,
        dia_quoting: string,
        dia_skipinitialspace: bool,
        dia_skiptrailingspace: bool,
        dia_lineterminator: string,
        dia_strict: bool,
        guess_dtypes: bool,
        tsk_pages: seq[string],
        tsk_offset: uint,
        tsk_count: int,
        import_fields: seq[uint]
    ): void {.exportpy.} =
        try:
            text_reader.toTaskArgs(
                path = path,
                encoding = encoding,
                dia_delimiter = dia_delimiter,
                dia_quotechar = dia_quotechar,
                dia_escapechar = dia_escapechar,
                dia_doublequote = dia_doublequote,
                dia_quoting = dia_quoting,
                dia_skipinitialspace = dia_skipinitialspace,
                dia_skiptrailingspace = dia_skiptrailingspace,
                dia_lineterminator = dia_lineterminator,
                dia_strict = dia_strict,
                guess_dtypes = guess_dtypes,
                tsk_pages = tsk_pages,
                tsk_offset = tsk_offset,
                tsk_count = uint tsk_count,
                import_fields = import_fields
            ).textReaderTask
        except Exception as e:
            echo $e.msg & "\n" & $e.getStackTrace
            raise e

    proc text_reader(
        pid: string, path: string, encoding: string,
        columns: PyObject, first_row_has_headers: bool, header_row_index: uint,
        start: PyObject, limit: PyObject,
        guess_datatypes: bool,
        newline: string, delimiter: string, text_qualifier: string,
        strip_leading_and_tailing_whitespace: bool,
        page_size: uint,
        quoting: string
    ): text_reader.TabliteTable {.exportpy.} =
        try:
            var arg_cols = (if isNone(columns): none[seq[string]]() else: some(columns.to(seq[string])))
            var arg_encoding = str2Enc(encoding)
            var arg_start = (if isNone(start): none[int]() else: some(start.to(int)))
            var arg_limit = (if isNone(limit): none[int]() else: some(limit.to(int)))
            var arg_newline = (if newline.len == 1: newline[0] else: raise newException(Exception, "'newline' not a char"))
            var arg_delimiter = (if delimiter.len == 1: delimiter[0] else: raise newException(Exception, "'delimiter' not a char"))
            var arg_text_qualifier = (if text_qualifier.len == 1: text_qualifier[0] else: raise newException(Exception, "'text_qualifier' not a char"))
            var arg_quoting = str2quoting(quoting)

            let table = textReader(
                    pid = pid,
                    path = path,
                    encoding = arg_encoding,
                    columns = arg_cols,
                    first_row_has_headers = first_row_has_headers,
                    header_row_index = header_row_index,
                    start = arg_start,
                    limit = arg_limit,
                    guess_datatypes = guess_datatypes,
                    newline = arg_newline,
                    delimiter = arg_delimiter,
                    text_qualifier = arg_text_qualifier,
                    strip_leading_and_tailing_whitespace = strip_leading_and_tailing_whitespace,
                    page_size = page_size,
                    quoting = arg_quoting
            )

            return table
        except Exception as e:
            echo $e.msg & "\n" & $e.getStackTrace
            raise e
    # --------   TEXT READER   --------
