import std/[os, options, strutils, osproc, sugar]
import encfile, table, csvparse, textreader, utils, pylayer, taskargs

# include pylib
import nimpy
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
        toTaskArgs(
            path=path,
            encoding=encoding,
            dia_delimiter=dia_delimiter,
            dia_quotechar=dia_quotechar,
            dia_escapechar=dia_escapechar,
            dia_doublequote=dia_doublequote,
            dia_quoting=dia_quoting,
            dia_skipinitialspace=dia_skipinitialspace,
            dia_skiptrailingspace=dia_skiptrailingspace,
            dia_lineterminator=dia_lineterminator,
            dia_strict=dia_strict,
            guess_dtypes=guess_dtypes,
            tsk_pages=tsk_pages,
            tsk_offset=tsk_offset,
            tsk_count=uint tsk_count,
            import_fields=import_fields
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
    guess_dtypes: bool,
    quoting: string
): TabliteTable {.exportpy.} =
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
                page_size=page_size,
                guess_dtypes=guess_dtypes,
                quoting=arg_quoting
        )

        discard table.task.saveTasks(pid)

        return table
    except Exception as e:
        echo $e.msg & "\n" & $e.getStackTrace
        raise e



if isMainModule:
    echo "Nimlite imported!"


proc runTask(path: string, encoding: string, dialect: TabliteDialect, task: TabliteTask, import_fields: seq[uint], guess_dtypes: bool): void =
    toTaskArgs(
        path=path,
        encoding=encoding,
        dia_delimiter=dialect.delimiter,
        dia_quotechar=dialect.quotechar,
        dia_escapechar=dialect.escapechar,
        dia_doublequote=dialect.doublequote,
        dia_quoting=dialect.quoting,
        dia_skipinitialspace=dialect.skipinitialspace,
        dia_skiptrailingspace=dialect.skiptrailingspace,
        dia_lineterminator=dialect.lineterminator,
        dia_strict=dialect.strict,
        guess_dtypes=guess_dtypes,
        tsk_pages=task.pages,
        tsk_offset=task.offset,
        tsk_count=task.count,
        import_fields=import_fields
    ).textReaderTask

proc executeParallel(path: string): void =
    echo "Executing tasks: '" & path & "'"
    let args = @[
        "--progress",
        "-a",
        "\"" & path & "\""
    ]

    let para = "/usr/bin/parallel"
    let ret_code = execCmd(para & " " & args.join(" "))

    if ret_code != 0:
        raise newException(Exception, "Process failed with errcode: " & $ret_code)

when isMainModule and appType != "lib":
    import argparse
    import std/times

    var path_csv: string
    var encoding: Encodings
    var dialect: Dialect
    var cols = none[seq[string]]()
    var guess_dtypes: bool
    var pid = "/media/ratchet/hdd/tablite/nim"
    var page_size = uint 1_000_000

    const boolean_true_choices = ["true", "yes", "t", "y"]
    # const boolean_false_choices = ["false", "no", "f", "n"]
    const boolean_choices = ["true", "false", "yes", "no", "t", "f", "y", "n"]

    var p = newParser:
        help("Imports tablite pages")
        option(
            "-e", "--encoding",
            help="file encoding",
            choices = @[$ENC_UTF8, $ENC_UTF16, $ENC_WIN1250],
            default=some($ENC_UTF8)
        )

        option(
            "-dt", "--guess_dtypes",
            help="gues datatypes",
            choices = @boolean_choices,
            default=some("true")
        )

        option("--delimiter", help="text delimiter", default=some(","))
        option("--quotechar", help="text quotechar", default=some("\""))
        option("--escapechar", help="text escapechar", default=some("\\"))
        option("--lineterminator", help="text lineterminator", default=some("\\n"))

        option(
            "--doublequote",
            help="text doublequote",
            choices = @boolean_choices,
            default=some("true")
        )

        option(
            "--skipinitialspace",
            help="text skipinitialspace",
            choices = @boolean_choices,
            default=some("false")
        )

        option(
            "--skiptrailingspace",
            help="text skiptrailingspace",
            choices = @boolean_choices,
            default=some("false")
        )

        option(
            "--quoting",
            help="text quoting",
            choices = @[
                $QUOTE_MINIMAL,
                $QUOTE_NONE,
            ],
            default=some($QUOTE_MINIMAL)
        )

        command("import"):
            arg("path", help="file path")
            arg("execute", help="execute immediatly")
            arg("multiprocess", help="use multiprocessing")
            run:
                path_csv = opts.path
        command("task"):
            option("--pages", help="task pages", required = true)
            option("--fields", help="fields to import", required = true)

            arg("path", help="file path")
            arg("offset", help="file offset")
            arg("count", help="line count")
            run:
                path_csv = opts.path
        run:
            var delimiter = opts.delimiter.unescapeSeq()
            var quotechar = opts.quotechar.unescapeSeq()
            var escapechar = opts.escapechar.unescapeSeq()
            var lineterminator = opts.lineterminator.unescapeSeq()

            if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
            if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
            if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
            if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

            dialect = newDialect(
                delimiter = delimiter[0],
                quotechar = quotechar[0],
                escapechar = escapechar[0],
                doublequote = opts.doublequote in boolean_true_choices,
                quoting = str2quoting(opts.quoting),
                skipinitialspace = opts.skipinitialspace in boolean_true_choices,
                skiptrailingspace = opts.skiptrailingspace in boolean_true_choices,
                lineterminator = lineterminator[0],
            )

            encoding = str2Enc(opts.encoding)

            guess_dtypes = opts.guess_dtypes in boolean_true_choices

    let opts = p.parse()
    p.run()

    if opts.import.isNone and opts.task.isNone:
        guess_dtypes = true
        (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/bad_empty.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/book1.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_test.csv", ENC_UTF16)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/win1250_test.csv", ENC_WIN1250)

        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/book1.txt", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/gdocs1.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M_1col.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", ENC_UTF16)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", ENC_UTF16)

        # cols = some(@["\"Item\"", "\"Materiál\"", "\"Objem\"", "\"Jednotka objemu\"", "\"Free Inv Pcs\""])
        # dialect.quoting = Quoting.QUOTE_NONE
        # dialect.delimiter = ';'

        let multiprocess = false
        let execute = true
        let d0 = getTime()

        let table = importTextFile(pid, path_csv, encoding, dialect, cols, false, 0, page_size, guess_dtypes, some[int](0), some[int](5))
        let task = table.task


        if multiprocess:
            let task_path = task.saveTasks(pid)

            if execute:
                executeParallel(task_path)
        else:
            if execute:
                for column_task in task.tasks:
                    runTask(task.path, task.encoding, task.dialect, column_task, task.import_fields, task.guess_dtypes)

        let d1 = getTime()
        echo $(d1 - d0)

    else:
        if opts.import.isSome:
            raise newException(Exception, "not implemented: 'import'")
        elif opts.task.isSome:
            let tdia = newTabliteDialect(dialect)
            let count = parseInt(opts.task.get.count)
            let ttask = newTabliteTask(opts.task.get.pages.split(","), uint parseInt(opts.task.get.offset), uint count)
            let fields = collect: (for k in opts.task.get.fields.split(","): uint parseInt(k))

            runTask(path_csv, $encoding, tdia, ttask, fields, guess_dtypes)

            # let fields = opts.task.get.fields.split(",")
            # let pages = 
            # toTaskArgs(
            #     path=path_csv,
            #     encoding= $encoding,
            #     dia_delimiter= $dialect.delimiter,
            #     dia_quotechar= $dialect.quotechar,
            #     dia_escapechar= $dialect.escapechar,
            #     dia_doublequote=dialect.doublequote,
            #     dia_quoting= $dialect.quoting,
            #     dia_skipinitialspace=dialect.skipinitialspace,
            #     dia_lineterminator= $dialect.lineterminator,
            #     dia_strict=dialect.strict,
            #     guess_dtypes=guess_dtypes,
            #     tsk_pages=opts.task.get.pages.split(","),
            #     tsk_offset=uint parseInt(opts.task.get.offset),
            #     import_fields=pages,
            #     count=parseInt(opts.task.get.count)
            # ).textReaderTask

# import std/[os, enumerate, sugar, times, tables, sequtils, json, unicode, osproc, options]
# import argparse
# import encfile, csvparse, table, utils, textreader

# if isMainModule:
#     var path_csv: string
#     var encoding: Encodings
#     var dialect: Dialect
#     var cols = none[seq[string]]()

#     const boolean_true_choices = ["true", "yes", "t", "y"]
#     # const boolean_false_choices = ["false", "no", "f", "n"]
#     const boolean_choices = ["true", "false", "yes", "no", "t", "f", "y", "n"]

#     var p = newParser:
#         help("Imports tablite pages")
#         option(
#             "-e", "--encoding",
#             help="file encoding",
#             choices = @["UTF8", "UTF16"],
#             default=some("UTF8")
#         )

#         option("--delimiter", help="text delimiter", default=some(","))
#         option("--quotechar", help="text quotechar", default=some("\""))
#         option("--escapechar", help="text escapechar", default=some("\\"))
#         option("--lineterminator", help="text lineterminator", default=some("\\n"))

#         option(
#             "--doublequote",
#             help="text doublequote",
#             choices = @boolean_choices,
#             default=some("true")
#         )

#         option(
#             "--skipinitialspace",
#             help="text skipinitialspace",
#             choices = @boolean_choices,
#             default=some("false")
#         )

#         option(
#             "--quoting",
#             help="text quoting",
#             choices = @[
#                 "QUOTE_MINIMAL",
#                 "QUOTE_ALL",
#                 "QUOTE_NONNUMERIC",
#                 "QUOTE_NONE",
#                 "QUOTE_STRINGS",
#                 "QUOTE_NOTNULL"
#             ],
#             default=some("QUOTE_MINIMAL")
#         )

#         command("import"):
#             arg("path", help="file path")
#             arg("execute", help="execute immediatly")
#             arg("multiprocess", help="use multiprocessing")
#             run:
#                 discard
#         command("task"):
#             option("--pages", help="task pages", required = true)
#             option("--fields_keys", help="field keys", required = true)
#             option("--fields_vals", help="field vals", required = true)

#             arg("path", help="file path")
#             arg("offset", help="file offset")
#             arg("count", help="line count")
#             run:
#                 discard
#         run:
#             var delimiter = opts.delimiter.unescapeSeq()
#             var quotechar = opts.quotechar.unescapeSeq()
#             var escapechar = opts.escapechar.unescapeSeq()
#             var lineterminator = opts.lineterminator.unescapeSeq()

#             if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
#             if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
#             if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
#             if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

#             dialect = newDialect(
#                 delimiter = delimiter[0],
#                 quotechar = quotechar[0],
#                 escapechar = escapechar[0],
#                 doublequote = opts.doublequote in boolean_true_choices,
#                 quoting = str2quoting(opts.quoting),
#                 skipinitialspace = opts.skipinitialspace in boolean_true_choices,
#                 lineterminator = lineterminator[0],
#             )

#             case opts.encoding.toUpper():
#                 of "UTF8": encoding = ENC_UTF8
#                 of "UTF16": encoding = ENC_UTF16
#                 else: raise newException(Exception, "invalid 'encoding'")

#     let opts = p.parse()
#     p.run()

#     if opts.import.isNone and opts.task.isNone:
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/bad_empty.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/gdocs1.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M_1col.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", ENC_UTF8)
#         (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", ENC_UTF16)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", ENC_UTF16)

#         let d0 = getTime()
#         # cols = some(@["B", "A", "B"])
#         echo $importTextFile(path_csv, encoding, dialect, cols, true, multiprocess=false)
#         let d1 = getTime()

#         echo $(d1 - d0)
#     else:
#         if opts.import.isSome:
#             let execute = opts.import.get.execute in boolean_true_choices
#             let multiprocess = opts.import.get.multiprocess in boolean_true_choices
#             let path_csv = opts.import.get.path
#             echo "Importing: '" & path_csv & "'"

#             let d0 = getTime()
#             discard importTextFile(path_csv, encoding, dialect, cols, execute, multiprocess)
#             let d1 = getTime()

#             echo $(d1 - d0)

#         if opts.task.isSome:
#             let path = opts.task.get.path
#             var pages = opts.task.get.pages.split(",")
#             let fields_keys = opts.task.get.fields_keys.split(",")
#             let fields_vals = opts.task.get.fields_vals.split(",")

#             var field_relation = collect(initOrderedTable()):
#                 for (k, v) in zip(fields_keys, fields_vals):
#                     {parseUInt(k): v}
#             let import_fields = collect: (for k in field_relation.keys: k)

#             let offset = parseUInt(opts.task.get.offset)
#             let count = parseInt(opts.task.get.count)

#             textReaderTask(path, encoding, dialect, pages, import_fields.unsafeAddr, offset, count)
