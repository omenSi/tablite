import argparse, os, std/enumerate, sugar, times, tables, sequtils, json, unicode, osproc
import encfile, csvparse, paging

proc text_reader_task(
    path: string, encoding: Encodings, dialect: Dialect, 
    destinations: var seq[string], import_fields: ptr seq[uint],
    row_offset: uint, row_count: int): void =
    var obj = newReaderObj(dialect)
    
    let fh = newFile(path, encoding)
    let guess_dtypes = true
    let n_pages = destinations.len

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        var (n_rows, longest_str, ranks) = collectPageInfo(
            obj=obj.unsafeAddr,
            fh=fh.unsafeAddr,
            guess_dtypes=guess_dtypes,
            n_pages=n_pages,
            row_count=row_count,
            import_fields=import_fields
        )

        var (page_file_handlers, column_dtypes, binput) = dumpPageHeader(
            destinations=destinations,
            n_pages=n_pages,
            n_rows=n_rows,
            guess_dtypes=guess_dtypes,
            longest_str=longest_str,
            ranks=ranks,
        )

        try:
            fh.setFilePos(int64 row_offset, fspSet)

            dumpPageBody(
                obj=obj.unsafeAddr,
                fh=fh.unsafeAddr,
                guess_dtypes=guess_dtypes,
                n_pages=n_pages,
                row_count=row_count,
                import_fields=import_fields,
                page_file_handlers=page_file_handlers,
                longest_str=longest_str,
                ranks=ranks,
                column_dtypes=column_dtypes,
                binput=binput
            )

            dumpPageFooter(
                n_pages=n_pages,
                n_rows=n_rows,
                page_file_handlers=page_file_handlers,
                column_dtypes=column_dtypes,
                binput=binput
            )
        finally:
            for f in page_file_handlers:
                f.close()

    finally:
        fh.close()

proc uniqueName(desired_name: string, name_list: seq[string]): string =
    var name = desired_name
    var idx = 1

    while name in name_list:
        name = desired_name & "_" & $idx
        inc idx

    return name

proc import_file(path: string, encoding: Encodings, dia: Dialect, columns: ptr seq[string], execute: bool, multiprocess: bool): void =
    echo "Collecting tasks: '" & path & "'"
    let (newline_offsets, newlines) = findNewlines(path, encoding)

    let dirname = "/media/ratchet/hdd/tablite/nim/page"

    if not dirExists(dirname):
        createDir(dirname)

    if newlines > 0:
        let fields = readColumns(path, encoding, dia, newline_offsets[0])

        var imp_columns: seq[string]

        if columns == nil:
            imp_columns = fields
        else:
            var missing = newSeq[string]()
            for column in columns[]:
                if not (column in fields):
                    missing.add("'" & column & "'")
            if missing.len > 0:
                raise newException(IOError, "Missing columns: [" & missing.join(", ") & "]")
            imp_columns = columns[]

        var field_relation = collect(initOrderedTable()):
            for ix, name in enumerate(fields):
                if name in imp_columns:
                    {uint ix: name}
        let import_fields = collect: (for k in field_relation.keys: k)

        var field_relation_inv = collect(initOrderedTable()):
            for (ix, name) in field_relation.pairs:
                {name: ix}

        var page_list = collect(initOrderedTable()):
            for (ix, name) in field_relation.pairs:
                {ix: newSeq[string]()}

        var name_list = newSeq[string]()
        var table_columns = collect(initOrderedTable()):
            for name in imp_columns:
                let unq = uniqueName(name, name_list)
                
                name_list.add(unq)

                {unq: field_relation_inv[name]}

        var page_idx: uint32 = 1
        var row_idx: uint = 1
        var page_size: uint = 1_000_000

        let path_task = dirname & "/tasks.txt"
        let ft = open(path_task, fmWrite)

        var delimiter = ""
        delimiter.addEscapedChar(dia.delimiter)
        var quotechar = ""
        quotechar.addEscapedChar(dia.quotechar)
        var escapechar = ""
        escapechar.addEscapedChar(dia.escapechar)
        var lineterminator = ""
        lineterminator.addEscapedChar(dia.lineterminator)

        echo "Dumping tasks: '" & path & "'"
        while row_idx < newlines:
            let page_count = field_relation.len
            var pages = newSeq[string](page_count)

            for idx in 0..page_count - 1:
                let pagepath =  dirname & "/" & $page_idx & ".npy"
                let field_idx = import_fields[idx]

                page_list[field_idx].add(pagepath)
                pages[idx] = pagepath

                inc page_idx

            if not multiprocess:
                text_reader_task(path, encoding, dia, pages, import_fields.unsafeAddr, newline_offsets[row_idx], int page_size)

            ft.write("\"" & getAppFilename() & "\" ")

            case encoding:
                of ENC_UTF8:
                    ft.write("--encoding=" & "UTF8" & " ")
                of ENC_UTF16:
                    ft.write("--encoding" & "UTF16" & " ")

            ft.write("--delimiter=\"" & delimiter & "\" ")
            ft.write("--quotechar=\"" & quotechar & "\" ")
            ft.write("--escapechar=\"" & escapechar & "\" ")
            ft.write("--lineterminator=\"" & lineterminator & "\" ")
            ft.write("--doublequote=" & $dia.doublequote & " ")
            ft.write("--skipinitialspace=" & $dia.skipinitialspace & " ")
            ft.write("--quoting=" & $dia.quoting & " ")

            ft.write("task ")

            ft.write("--pages=\"" & pages.join(",") & "\" ")
            ft.write("--fields_keys=\"" & toSeq(field_relation.keys).join(",") & "\" ")
            ft.write("--fields_vals=\"" & toSeq(field_relation.values).join(",") & "\" ")

            ft.write("\"" & path & "\" ")
            ft.write($newline_offsets[row_idx] & " ")
            ft.write($page_size)

            ft.write("\n")

            row_idx = row_idx + page_size

        ft.close()

        var table_layout = collect(newSeq(imp_columns.len)):
            for 

        if multiprocess and execute:
            echo "Executing tasks: '" & path & "'"
            let args = @[
                "--progress",
                "-a",
                "\"" & path_task & "\""
            ]

            let para = "/usr/bin/parallel"

            let ret_code = execCmd(para & " " & args.join(" "))

            if ret_code != 0:
                raise newException(Exception, "Process failed with errcode: " & $ret_code)
    else:
        raise newException(IOError, "end of file")

proc unescape_seq(str: string): string = # nim has no true unescape
    case str:
        of "\\n": return "\n"
        of "\\t": return "\t"

    return str

if isMainModule:
    var path_csv: string
    var encoding: Encodings
    var dialect: Dialect

    const boolean_true_choices = ["true", "yes", "t", "y"]
    # const boolean_false_choices = ["false", "no", "f", "n"]
    const boolean_choices = ["true", "false", "yes", "no", "t", "f", "y", "n"]

    var p = newParser:
        help("Imports tablite pages")
        option(
            "-e", "--encoding",
            help="file encoding",
            choices = @["UTF8", "UTF16"],
            default=some("UTF8")
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
            "--quoting",
            help="text quoting",
            choices = @[
                "QUOTE_MINIMAL",
                "QUOTE_ALL",
                "QUOTE_NONNUMERIC",
                "QUOTE_NONE",
                "QUOTE_STRINGS",
                "QUOTE_NOTNULL"
            ],
            default=some("QUOTE_MINIMAL")
        )

        command("import"):
            arg("path", help="file path")
            arg("execute", help="execute immediatly")
            arg("multiprocess", help="use multiprocessing")
            run:
                discard
        command("task"):
            option("--pages", help="task pages", required = true)
            option("--fields_keys", help="field keys", required = true)
            option("--fields_vals", help="field vals", required = true)

            arg("path", help="file path")
            arg("offset", help="file offset")
            arg("count", help="line count")
            run:
                discard
        run:
            var delimiter = opts.delimiter.unescape_seq()
            var quotechar = opts.quotechar.unescape_seq()
            var escapechar = opts.escapechar.unescape_seq()
            var lineterminator = opts.lineterminator.unescape_seq()

            if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
            if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
            if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
            if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

            dialect = newDialect(
                delimiter = delimiter[0],
                quotechar = quotechar[0],
                escapechar = escapechar[0],
                doublequote = opts.doublequote in boolean_true_choices,
                quoting = (
                    case opts.quoting.toUpper():
                        of "QUOTE_MINIMAL":
                            QUOTE_MINIMAL
                        of "QUOTE_ALL":
                            QUOTE_ALL
                        of "QUOTE_NONNUMERIC":
                            QUOTE_NONNUMERIC
                        of "QUOTE_NONE":
                            QUOTE_NONE
                        of "QUOTE_STRINGS":
                            QUOTE_STRINGS
                        of "QUOTE_NOTNULL":
                            QUOTE_NOTNULL
                        else:
                            raise newException(Exception, "invalid 'quoting'")
                ),
                skipinitialspace = opts.skipinitialspace in boolean_true_choices,
                lineterminator = lineterminator[0],
            )

            case opts.encoding.toUpper():
                of "UTF8": encoding = ENC_UTF8
                of "UTF16": encoding = ENC_UTF16
                else: raise newException(Exception, "invalid 'encoding'")

    let opts = p.parse()
    p.run()

    if opts.import.isNone and opts.task.isNone:
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/bad_empty.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/gdocs1.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M_1col.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", ENC_UTF8)
        (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", ENC_UTF16)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", ENC_UTF16)

        let d0 = getTime()
        let cols = @["B", "A", "B"]
        import_file(path_csv, encoding, dialect, cols.unsafeAddr, true, multiprocess=false)
        let d1 = getTime()
        
        echo $(d1 - d0)
    else:
        if opts.import.isSome:
            let execute = opts.import.get.execute in boolean_true_choices
            let multiprocess = opts.import.get.multiprocess in boolean_true_choices
            let path_csv = opts.import.get.path
            echo "Importing: '" & path_csv & "'"
            
            let d0 = getTime()
            import_file(path_csv, encoding, dialect, nil, execute, multiprocess)
            let d1 = getTime()
            
            echo $(d1 - d0)

        if opts.task.isSome:
            let path = opts.task.get.path
            var pages = opts.task.get.pages.split(",")
            let fields_keys = opts.task.get.fields_keys.split(",")
            let fields_vals = opts.task.get.fields_vals.split(",")

            var field_relation = collect(initOrderedTable()):
                for (k, v) in zip(fields_keys, fields_vals):
                    {parseUInt(k): v}
            let import_fields = collect: (for k in field_relation.keys: k)

            let offset = parseUInt(opts.task.get.offset)
            let count = parseInt(opts.task.get.count)

            text_reader_task(path, encoding, dialect, pages, import_fields.unsafeAddr, offset, count)