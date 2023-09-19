import argparse
import std/enumerate
import os, math, sugar, times, tables, sequtils, json, unicode, parseutils, encodings, bitops, osproc, lists, endians

const NOT_SET = uint32.high
const EOL = uint32.high - 1

type Encodings {.pure.} = enum ENC_UTF8, ENC_UTF16

type BaseEncodedFile = ref object of RootObj
    fh: File

type FileUTF8 = ref object of BaseEncodedFile
type FileUTF16 = ref object of BaseEncodedFile
    endianness: Endianness

type DataTypes = enum
    DT_INT, DT_BOOLEAN, DT_FLOAT,
    DT_DATE, DT_TIME,
    DT_STRING,
    DT_DATETIME,
    DT_MAX_ELEMENTS

type PY_NoneType = object
let PY_None = PY_NoneType()

type PY_Date = object
    year: uint16
    month, day: uint8

type PY_Time = object
    hour, minute, second: uint8
    microsecond: uint32
    has_tz: bool
    tz_days, tz_seconds, tz_microseconds: int32

proc newPyDate(d: DateTime): PY_Date =
    let year = uint16 d.year
    let month = uint8 d.month
    let day = uint8 d.monthday
    return PY_Date(year: year, month: month, day: day)

proc newPyTime(hour: uint8, minute: uint8, second: uint8, microsecond: uint32): PY_Time =
    return PY_Time(hour: hour, minute: minute, second: second, microsecond: microsecond)

proc newPyTime(hour: uint8, minute: uint8, second: uint8, microsecond: uint32, tz_days: int32, tz_seconds: int32, tz_microseconds: int32): PY_Time =
    if tz_days == 0 and tz_seconds == 0:
        return newPyTime(hour, minute, second, microsecond)
    
    return PY_Time(
            hour: hour, minute: minute, second: second, microsecond: microsecond,
            has_tz: true,
            tz_days: tz_days, tz_seconds: tz_seconds, tz_microseconds: tz_microseconds
        )


proc parse_int(str: ptr string): int = parseInt(str[])
proc parse_float(str: ptr string): float = parseFloat(str[])

proc parse_bool(str: ptr string): bool =
    if str[].toLower() == "true":
        return true
    elif str[].toLower() == "false":
        return false

    raise newException(ValueError, "not a boolean value")

const ValidDateFormats = @[
    "yyyy-MM-d",
    "yyyy-M-d",
    "yyyy-MM-N",
    "yyyy-M-N",
    "d-MM-yyyy",
    "N-MM-yyyy",
    "d-M-yyyy",
    "N-N-yyyy",
    "!yyyy-MM-d", # nim doesn't support dot types, check if starts with '!' and replace with other '-'
    "!yyyy-M-d",
    "!yyyy-MM-N",
    "!yyyy-M-N",
    "!d.MM.yyyy",
    "!N.MM.yyyy",
    "!d.M.yyyy",
    "!N.M.yyyy",
    "yyyy/MM/d",
    "yyyy/M/d",
    "yyyy/MM/N",
    "yyyy/M/N",
    "d/MM/yyyy",
    "N/MM/yyyy",
    "d/M/yyyy",
    "N/M/yyyy",
    "yyyy MM d",
    "yyyy M d",
    "yyyy MM N",
    "yyyy M N",
    "d MM yyyy",
    "N M yyyy",
    "d M yyyy",
    "N MM yyyy",
    "yyyyMMdd",
]


proc parse_date(str: ptr string): PY_Date =
    echo $str[]
    for fmt in ValidDateFormats:
        if fmt[0] == '!':
            let replaced_str = str[].replace(".", "-")
            let replaced_fmt = fmt.substr(1)
            
            try:
                return newPyDate(parse(replaced_str, replaced_fmt))
            except ValueError:
                continue

        try:
            return newPyDate(parse(str[], fmt))
        except ValueError:
            continue

    raise newException(ValueError, "not a date")

# # Format supported is HH[:MM[:SS[.fff[fff]]]][+HH:MM[:SS[.ffffff]]]
# const ValidTimeFormats = @[
#     initTimeFormat("HH"),
#     initTimeFormat("HH:mm"),
#     initTimeFormat("HH:mm:ss"),
#     initTimeFormat("HH:mm:ss.fff"),
#     initTimeFormat("HH:mm:ss.ffffff"),

#     initTimeFormat("HH+HH:mm"),
#     initTimeFormat("HH+HH:mm:ss"),
#     initTimeFormat("HH+HH:mm:ss.ffffff"),
    
#     initTimeFormat("HH:mm"),
#     initTimeFormat("HH:mm+HH:mm"),
#     initTimeFormat("HH:mm+HH:mm:ss"),
#     initTimeFormat("HH:mm+HH:mm:ss.ffffff"),

#     initTimeFormat("HH:mm:ss"),
#     initTimeFormat("HH:mm:ss+HH:mm"),
#     initTimeFormat("HH:mm:ss+HH:mm:ss"),
#     initTimeFormat("HH:mm:ss+HH:mm:ss.ffffff"),

#     initTimeFormat("HH:mm:ss.fff"),
#     initTimeFormat("HH:mm:ss.fff+HH:mm"),
#     initTimeFormat("HH:mm:ss.fff+HH:mm:ss"),
#     initTimeFormat("HH:mm:ss.fff+HH:mm:ss.fffffff"),

#     initTimeFormat("HH:mm:ss.ffffff"),
#     initTimeFormat("HH:mm:ss.ffffff+HH:mm"),
#     initTimeFormat("HH:mm:ss.ffffff+HH:mm:ss"),
#     initTimeFormat("HH:mm:ss.ffffff+HH:mm:ss.ffffff"),
# ]

proc divmod(x: int, y: int): (int, int) =
    let z = int(floor(x / y))

    return (z, x - y * z)

proc to_timedelta(
    weeks = 0, days = 0, hours = 0, minutes = 0, seconds = 0, milliseconds = 0, microseconds: int = 0
): (int, int, int) =
    var d, s, us: int

    var v_weeks = weeks
    var v_days = days
    var v_hours = hours
    var v_minutes = minutes
    var v_seconds = seconds
    var v_milliseconds = milliseconds
    var v_microseconds = microseconds

    # Normalize everything to days, seconds, microseconds.
    v_days += v_weeks*7
    v_seconds += v_minutes*60 + v_hours*3600
    v_microseconds += v_milliseconds*1000

    d = v_days

    (v_days, v_seconds) = divmod(v_seconds, 24*3600)

    d += v_days
    s += int(v_seconds)    # can't overflow

    v_microseconds = int(v_microseconds)
    (v_seconds, v_microseconds) = divmod(v_microseconds, 1000000)
    (v_days, v_seconds) = divmod(v_seconds, 24*3600)
    d += v_days
    s += v_seconds

    # Just a little bit of carrying possible for microseconds and seconds.
    (v_seconds, us) = divmod(v_microseconds, 1000000)
    s += v_seconds
    (v_days, s) = divmod(s, 24*3600)
    d += v_days


    return (d, s, us)

proc parse_hh_mm_ss_ff(tstr: ptr string): (uint8, uint8, uint8, uint32) =
    # Parses things of the form HH[:MM[:SS[.fff[fff]]]]
    let len_str = tstr[].len

    var time_comps: array[4, int]
    var pos = 0

    for comp in 0..2:
        if (len_str - pos) < 2:
            raise newException(ValueError, "Incomplete time component")

        let substr = tstr[].substr(pos, pos+1)

        time_comps[comp] = parseInt(substr)

        pos += 2

        if pos >= len_str or comp >= 2:
            break
        
        let next_char = tstr[pos]

        if next_char != ':':
            raise newException(ValueError, "Invalid time separator: " & $next_char)

        pos += 1

    if pos < len_str:
        if tstr[pos] != '.':
            raise newException(ValueError, "Invalid microsecond component")
        else:
            pos += 1

            let len_remainder = len_str - pos
            if not (len_remainder in [3, 6]):
                raise newException(ValueError, "Invalid microsecond component")

            time_comps[3] = parseInt(tstr[].substr(pos))
            if len_remainder == 3:
                time_comps[3] *= 1000

    return (uint8 time_comps[0], uint8 time_comps[1], uint8 time_comps[2], uint32 time_comps[3])

proc parse_time(str: ptr string): PY_Time =
    # Format supported is HH[:MM[:SS[.fff[fff]]]][+HH:MM[:SS[.ffffff]]]
    if not (":" in str[]) or str[].len < 2:
        raise newException(ValueError, "not a time")

    let tz_pos_minus = str[].find("-")
    let tz_pos_plus = str[].find("+")

    var tz_pos = -1

    if tz_pos_minus != -1 and tz_pos_plus != -1:
        raise newException(ValueError, "not a time")
    elif tz_pos_plus != -1:
        tz_pos = tz_pos_plus
    elif tz_pos_minus != -1:
        tz_pos = tz_pos_minus

    var timestr = (if tz_pos == -1: str[] else: str[].substr(0, tz_pos-1))

    let (hour, minute, second, microsecond) = parse_hh_mm_ss_ff(timestr.unsafeAddr)

    if tz_pos >= 0:
        let tzstr = str[].substr(tz_pos + 1)

        if not (tzstr.len in [5, 8, 15]):
            raise newException(Exception, "invalid timezone")

        echo $str[tz_pos]

        let tz_sign: int8 = (if str[tz_pos] == '-': -1 else: 1)
        let (tz_hours_p, tz_minutes_p, tz_seconds_p, tz_microseconds_p) = parse_hh_mm_ss_ff(tzstr.unsafeAddr)
        var (tz_days, tz_seconds, tz_microseconds) = to_timedelta(
            hours = tz_sign * int tz_hours_p,
            minutes = tz_sign * int tz_minutes_p,
            seconds = tz_sign * int tz_seconds_p,
            microseconds = tz_sign * int tz_microseconds_p
        )

        return newPyTime(hour, minute, second, microsecond, int32 tz_days, int32 tz_seconds, int32 tz_microseconds)
    
    return newPyTime(hour, minute, second, microsecond)

proc parse_datetime(str: ptr string): DateTime =
    raise newException(Exception, "not implemented error: parse_datetime (" & $str[] & ")")

type Rank = array[int(DataTypes.DT_MAX_ELEMENTS), (DataTypes, uint)]

iterator iter(rank: var Rank): ptr (DataTypes, uint) {.closure.} =
    var x = 0
    let max = int(DataTypes.DT_MAX_ELEMENTS)
    while x < max:
        yield rank[x].unsafeAddr
        inc x

proc newRank(): Rank =
    var ranks {.noinit.}: Rank

    for i in 0..(int(DataTypes.DT_MAX_ELEMENTS)-1):
        ranks[i] = (DataTypes(i), uint 0)

    return ranks

proc endOfFile(f: BaseEncodedFile): bool = f.fh.endOfFile()
proc getFilePos(f: BaseEncodedFile): uint = uint f.fh.getFilePos()
proc setFilePos(f: BaseEncodedFile, pos: int64, relativeTo: FileSeekPos): void = f.fh.setFilePos(pos, relativeTo)
proc close(f: BaseEncodedFile): void = f.fh.close()

proc readLine(f: FileUTF8, str: var string): bool = f.fh.readLine(str)
proc readLine(f: FileUTF16, str: var string): bool = 
    var ch_arr {.noinit.}: array[2, uint8]
    var ch: uint16

    let newline_char: uint16 = 0x000a
    var wchar_seq {.noinit.} = newSeqOfCap[uint16](80)

    while unlikely(not f.endOfFile):
        if f.fh.readBuffer(addr ch_arr, 2) != ch_arr.len:
            raise newException(Exception, "malformed file")

        if f.endianness == bigEndian: # big if true
            (ch_arr[0], ch_arr[1]) = (ch_arr[1], ch_arr[0])

        ch = cast[uint16](ch_arr)

        if newline_char == ch:
            break

        wchar_seq.add(ch)

    var wstr {.noinit.} = newWideCString(wchar_seq.len)

    if wchar_seq.len > 0:
        copyMem(wstr[0].unsafeAddr, wchar_seq[0].unsafeAddr, wchar_seq.len * 2)
    else:
        return false

    str = $wstr

    return true

proc readLine(f: BaseEncodedFile, str: var string): bool = 
    if f of FileUTF8:
        return readLine(cast[FileUTF8](f), str)
    elif f of FileUTF16:
        return readLine(cast[FileUTF16](f), str)
    else:
        raise newException(Exception, "encoding not implemented")

proc newFileUTF16(filename: string): FileUTF16 =
    var fh = open(filename, fmRead)

    if fh.getFileSize() mod 2 != 0:
        raise newException(Exception, "invalid size")

    var bom_bytes: array[2, uint16]
    
    if fh.readBuffer(addr bom_bytes, bom_bytes.len) != bom_bytes.len:
        raise newException(Exception, "cannot find bom")

    var bom = cast[uint16](bom_bytes)
    var endianness: Endianness;

    if bom == 0xfeff:
        endianness = Endianness.littleEndian
    elif bom == 0xfffe:
        endianness = Endianness.bigEndian
    else:
        raise newException(Exception, "invalid bom")

    return FileUTF16(fh: fh, endianness: endianness)

proc newFile(filename: string, encoding: Encodings): BaseEncodedFile =
    case encoding:
        of ENC_UTF8:
            return FileUTF8(fh: open(filename, fmRead))
        of ENC_UTF16:
            return newFileUTF16(filename)
        else:
            raise newException(Exception, "encoding not implemented")

proc find_newlines(fh: BaseEncodedFile): (seq[uint], uint) =
    var newline_offsets = newSeq[uint](1)
    var total_lines: uint = 0
    var str: string

    newline_offsets[0] = fh.getFilePos()

    while likely(fh.readLine(str)):
        inc total_lines

        newline_offsets.add(fh.getFilePos())

    return (newline_offsets, total_lines)

proc find_newlines(path: string, encoding: Encodings): (seq[uint], uint) =
    let fh = newFile(path, encoding)
    try:
        return find_newlines(fh)
    finally:
        fh.close()

type Quoting {.pure.} = enum
    QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE,
    QUOTE_STRINGS, QUOTE_NOTNULL

type ParserState {.pure.} = enum
    START_RECORD, START_FIELD, ESCAPED_CHAR, IN_FIELD,
    IN_QUOTED_FIELD, ESCAPE_IN_QUOTED_FIELD, QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL, AFTER_ESCAPED_CRNL

type Dialect = object
    delimiter: char
    quotechar: char
    escapechar: char
    doublequote: bool
    quoting: Quoting
    skipinitialspace: bool
    lineterminator: char
    strict: bool

proc newDialect(delimiter: char = ',', quotechar: char = '"', escapechar: char = '\\', doublequote: bool = true, quoting: Quoting = QUOTE_MINIMAL, skipinitialspace: bool = false, lineterminator: char = '\n'): Dialect =
    Dialect(delimiter:delimiter, quotechar:quotechar, escapechar:escapechar, doublequote:doublequote, quoting:quoting, skipinitialspace:skipinitialspace, lineterminator:lineterminator)

const field_limit: uint = 128 * 1024;

type ReaderObj = object
    numeric_field: bool
    line_num: uint
    dialect: Dialect

    field_len: uint
    field_size: uint
    field: seq[char]

    fields: seq[string]
    field_count: uint

var readerAlloc = newSeq[string](1024)

proc newReaderObj(dialect: Dialect): ReaderObj =
    ReaderObj(dialect: dialect, fields: readerAlloc)

proc parse_grow_buff(self: var ReaderObj): bool =
    let field_size_new: uint = (if self.field_size > 0: 2u * self.field_size else: 4096u)
    
    self.field.setLen(field_size_new)
    self.field_size = field_size_new

    return true

proc parse_add_char(self: var ReaderObj, state: var ParserState, c: char): bool =
    if self.field_len >= field_limit:
        return false

    if unlikely(self.field_len == self.field_size and not self.parse_grow_buff()):
        return false

    self.field[self.field_len] = c
    inc self.field_len

    return true

proc parse_save_field(self: var ReaderObj): bool =
    if self.numeric_field:
        self.numeric_field = false

        raise newException(Exception, "not yet implemented: parse_save_field numeric_field")

    var field {.noinit.} = newString(self.field_len)

    if likely(self.field_len > 0):
        copyMem(field[0].unsafeAddr, self.field[0].unsafeAddr, self.field_len)

    if unlikely(self.field_count + 1 >= (uint self.field.high)):
        self.field.setLen(self.field.len() * 2)

    self.fields[self.field_count] = field

    inc self.field_count

    self.field_len = 0

    return true

proc parse_process_char(self: var ReaderObj, state: var ParserState, cc: uint32): bool =
    let dia = self.dialect
    var ch_code = cc
    var c = (if ch_code < EOL: char ch_code else: '\x00')

    case state:
        of START_RECORD, START_FIELD:
            if ch_code == EOL:
                return true

            if state == START_RECORD: # nim cannot fall through
                if unlikely(c in ['\n', '\r']):
                    state = EAT_CRNL
                else:
                    state = START_FIELD

            if unlikely(c in ['\n', '\r'] or unlikely(ch_code == EOL)):
                if unlikely(not self.parse_save_field()):
                    return false

                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif unlikely(c == dia.quotechar and dia.quoting != QUOTE_NONE):
                state = IN_QUOTED_FIELD
            elif unlikely(c == dia.escapechar):
                state = ESCAPED_CHAR
            elif unlikely(c == ' ' and dia.skipinitialspace):
                discard
            elif unlikely(c == dia.delimiter):
                if unlikely(not self.parse_save_field()):
                    return false
            else:
                if dia.quoting == QUOTE_NONNUMERIC:
                    self.numeric_field = true
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = IN_FIELD
        of ESCAPED_CHAR:
            if c in ['\n', '\r']:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = AFTER_ESCAPED_CRNL

            if ch_code == EOL:
                c = '\n'
                ch_code = uint32 c

            if unlikely(not self.parse_add_char(state, c)):
                return false

            state = IN_FIELD
        of AFTER_ESCAPED_CRNL, IN_FIELD:
            if state == AFTER_ESCAPED_CRNL and ch_code == EOL:
                return true # nim is stupid

            if unlikely(c in ['\n', '\r'] or unlikely(ch_code == EOL)):
                if unlikely(not self.parse_save_field()):
                    return false
                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif c == dia.escapechar:
                state = ESCAPED_CHAR
            elif c == dia.delimiter:
                if unlikely(not self.parse_save_field()):
                    return false
                state = START_FIELD
            else:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
        of IN_QUOTED_FIELD:
            if ch_code == EOL:
                discard
            elif c == dia.escapechar:
                state = ESCAPE_IN_QUOTED_FIELD
            elif c == dia.quotechar and dia.quoting != QUOTE_NONE:
                if dia.doublequote:
                    state = QUOTE_IN_QUOTED_FIELD
                else:
                    state = IN_FIELD
            else:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
        of ESCAPE_IN_QUOTED_FIELD:
            if ch_code == EOL:
                c = '\n'
                ch_code = uint32 c
            
            if unlikely(not self.parse_add_char(state, c)):
                return false

            state = IN_QUOTED_FIELD
        of QUOTE_IN_QUOTED_FIELD:
            if dia.quoting != QUOTE_NONE and c == dia.quotechar:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = IN_QUOTED_FIELD
            elif c == dia.delimiter:
                if unlikely(not self.parse_save_field()):
                    return false
                state = START_FIELD
            elif c in ['\n', '\r'] or ch_code == EOL:
                if unlikely(not self.parse_save_field()):
                    return false
                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif not dia.strict:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = IN_FIELD
            else:
                return false
        of EAT_CRNL:
            if c in ['\n', '\r']:
                discard
            elif ch_code == EOL:
                state = START_RECORD
            else:
                return false

    return true

iterator parse_csv(self: var ReaderObj, fh: BaseEncodedFile): (uint, ptr seq[string], uint) =
    let dia = self.dialect

    var state: ParserState = START_RECORD
    var line_num: uint = 0
    var line = newStringOfCap(80)
    var pos: uint
    var linelen: uint;

    self.field_len = 0
    self.field_count = 0

    while likely(not fh.endOfFile):
        if not fh.readLine(line):
            break

        if self.field_len != 0 and state == IN_QUOTED_FIELD:
            if dia.strict:
                raise newException(Exception, "unexpected end of data")
            elif self.parse_save_field():
                break

        line.add('\n')

        
        linelen = uint line.len
        pos = 0

        while pos < linelen:
            if unlikely(not self.parse_process_char(state, uint32 line[pos])):
                raise newException(Exception, "illegal")
            
            inc pos

        if unlikely(not self.parse_process_char(state, EOL)):
            raise newException(Exception, "illegal")

        yield (line_num, addr self.fields, self.field_count)

        self.field_count = 0

        inc line_num

iterator parse_csv(self: var ReaderObj, path: string, encoding: Encodings): (uint, ptr seq[string], uint) =
    var fh = newFile(path, encoding)

    try:
        for it in self.parse_csv(fh):
            yield it
    finally:
        fh.close()

proc read_columns(path: string, encoding: Encodings, dialect: Dialect, row_offset: uint): seq[string] =
    let fh = newFile(path, encoding)
    var obj = newReaderObj(dialect)

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        for (row_idx, fields, field_count) in obj.parse_csv(fh):
            return fields[0..field_count-1]
    finally:
        fh.close()

proc write_numpy_header(fh: File, dtype: string, shape: uint): void =
    const magic = "\x93NUMPY"
    const major = "\x01"
    const minor = "\x00"
    
    let header = "{'descr': '" & dtype & "', 'fortran_order': False, 'shape': (" & $shape & ",)}"
    let header_len = len(header)
    let padding = (64 - ((len(magic) + len(major) + len(minor) + 2 + header_len)) mod 64)
    
    let padding_header = uint16 (padding + header_len)

    fh.write(magic)
    fh.write(major)
    fh.write(minor)

    discard fh.writeBuffer(padding_header.unsafeAddr, 2)

    fh.write(header)

    for i in 0..padding-2:
        fh.write(" ")
    fh.write("\n")

proc `<` (a: (DataTypes, uint), b: (DataTypes, uint)): bool = a[1] < b[1]
proc `>` (a: (DataTypes, uint), b: (DataTypes, uint)): bool = a[1] > b[1]

proc insert_sort[T](a: var openarray[T]) =
    # our array is likely to be nearly sorted or already sorted, therefore the complexity is better than bubble sort
    for i in 1 .. a.high:
        let value = a[i]
        var j = i
        while j > 0 and value > a[j-1]:
            a[j] = a[j-1]
            dec j
        a[j] = value

proc update_rank(rank: var Rank, str: ptr string): (bool, DataTypes) =
    var rank_dtype: DataTypes
    var index: int
    var rank_count: uint
    var is_none: bool = false

    for i, r_addr in enumerate(rank.iter()):
        try:
            case r_addr[0]:
                of DataTypes.DT_INT:
                    discard str.parse_int()
                of DataTypes.DT_FLOAT:
                    discard str.parse_float()
                of DataTypes.DT_BOOLEAN:
                    discard str.parse_bool()
                of DataTypes.DT_DATE:
                    discard str.parse_date()
                of DataTypes.DT_TIME:
                    discard str.parse_time()
                of DataTypes.DT_DATETIME:
                    discard str.parse_datetime()
                of DataTypes.DT_STRING:
                    if str[] in ["null", "Null", "NULL", "#N/A", "#n/a", "", "None"]:
                        is_none = true
                else:
                    raise newException(Exception, "invalid type")
        except ValueError:
            continue

        rank_dtype = r_addr[0]
        rank_count = r_addr[1]
        index = i

        break

    if is_none:
        return (true, rank_dtype)

    rank[index] = (rank_dtype, rank_count + 1)
    rank.insert_sort()

    return (false, rank_dtype)

const PKL_BINPUT = 'q'
const PKL_LONG_BINPUT = 'r'
const PKL_TUPLE1 = '\x85'
const PKL_TUPLE2 = '\x86'
const PKL_TUPLE3 = '\x87'
const PKL_TUPLE = 't'
const PKL_PROTO = '\x80'
const PKL_GLOBAL = 'c'
const PKL_BININT1 = 'K'
const PKL_BININT2 = 'M'
const PKL_BININT = 'J'
const PKL_SHORT_BINBYTES = 'C'
const PKL_REDUCE = 'R'
const PKL_MARK = '('
const PKL_BINUNICODE = 'X'
const PKL_NEWFALSE = '\x89'
const PKL_NEWTRUE = '\x88'
const PKL_NONE = 'N'
const PKL_BUILD = 'b'
const PKL_EMPTY_LIST = ']'
const PKL_STOP = '.'
const PKL_APPENDS = 'e'
const PKL_BINFLOAT = 'G'

proc write_pickle_binput(fh: ptr File, binput: var uint32): void =
    if binput <= 0xff:
        fh[].write(PKL_BINPUT)
        discard fh[].writeBuffer(binput.unsafeAddr, 1)
        inc binput
        return
    
    fh[].write(PKL_LONG_BINPUT)
    discard fh[].writeBuffer(binput.unsafeAddr, 4)
    inc binput

proc write_pickle_global(fh: ptr File, module_name: string, import_name: string): void = 
    fh[].write(PKL_GLOBAL)

    fh[].write(module_name)
    fh[].write('\x0A')

    fh[].write(import_name)
    fh[].write('\x0A')

proc write_pickle_proto(fh: ptr File): void =
    fh[].write(PKL_PROTO)
    fh[].write("\3")

proc write_pickle_binint_generic[T:uint8|uint16|uint32](fh: ptr File, value: T): void =
    when T is uint8:
        fh[].write(PKL_BININT1)
        discard fh[].writeBuffer(value.unsafeAddr, 1)
    when T is uint16:
        fh[].write(PKL_BININT2)
        discard fh[].writeBuffer(value.unsafeAddr, 2)
    when T is uint32:
        fh[].write(PKL_BININT)
        discard fh[].writeBuffer(value.unsafeAddr, 4)

proc write_pickle_binfloat(fh: ptr File, value: float): void =
    # pickle stores floats big-endian
    var f: float

    f.unsafeAddr.bigEndian64(value.unsafeAddr)

    echo $value
    fh[].write(PKL_BINFLOAT)
    discard fh[].writeBuffer(f.unsafeAddr, 8)

proc write_pickle_binint[T: int|uint|int32|uint32](fh: ptr File, value: T): void =
    when T is int or T is int32:
        if value < 0:
            fh.write_pickle_binint_generic(uint32 value)
            return

    if value <= 0xff:
        fh.write_pickle_binint_generic(uint8 value)
        return

    if value <= 0xffff:
        fh.write_pickle_binint_generic(uint16 value)
        return

    fh.write_pickle_binint_generic(uint32 value)

proc write_pickle_shortbinbytes(fh: ptr File, value: string): void =
    fh[].write(PKL_SHORT_BINBYTES)

    let len = value.len()

    discard fh[].writeBuffer(len.unsafeAddr, 1)
    discard fh[].writeBuffer(value[0].unsafeAddr, value.len)

proc write_pickle_binunicode(fh: ptr File, value: string): void =
    let len = uint32 value.len
    
    fh[].write(PKL_BINUNICODE)
    discard fh[].writeBuffer(len.unsafeAddr, 4)
    discard fh[].writeBuffer(value[0].unsafeAddr, len)

proc write_pickle_boolean(fh: ptr File, value: bool): void =
    if value == true:
        fh[].write(PKL_NEWTRUE)
    else:
        fh[].write(PKL_NEWFALSE)

proc write_pickle_start(fh: ptr File, binput: var uint32, elem_count: uint): void =
    binput = 0

    fh.write_pickle_proto()
    fh.write_pickle_global("numpy.core.multiarray", "_reconstruct")
    fh.write_pickle_binput(binput)
    fh.write_pickle_global("numpy", "ndarray")
    fh.write_pickle_binput(binput)
    fh.write_pickle_binint(0)
    fh[].write(PKL_TUPLE1)
    fh.write_pickle_binput(binput)
    fh.write_pickle_shortbinbytes("b")
    fh.write_pickle_binput(binput)
    fh[].write(PKL_TUPLE3)
    fh.write_pickle_binput(binput)
    fh[].write(PKL_REDUCE)
    fh.write_pickle_binput(binput)
    fh[].write(PKL_MARK)

    if true:
        fh.write_pickle_binint(1)
        fh.write_pickle_binint(elem_count)
        fh[].write(PKL_TUPLE1)
        fh.write_pickle_binput(binput)
        fh.write_pickle_global("numpy", "dtype")
        fh.write_pickle_binput(binput)
        fh.write_pickle_binunicode("O8")
        fh.write_pickle_binput(binput)
        fh.write_pickle_boolean(false)
        fh.write_pickle_boolean(true)
        fh[].write(PKL_TUPLE3)
        fh.write_pickle_binput(binput)
        fh[].write(PKL_REDUCE)
        fh.write_pickle_binput(binput)
        fh[].write(PKL_MARK)

        if true:
            fh.write_pickle_binint(3)
            fh.write_pickle_binunicode("|")
            fh.write_pickle_binput(binput)
            fh[].write(PKL_NONE)
            fh[].write(PKL_NONE)
            fh[].write(PKL_NONE)
            fh.write_pickle_binint(-1)
            fh.write_pickle_binint(-1)
            fh.write_pickle_binint(63)
            fh[].write(PKL_TUPLE)

        fh.write_pickle_binput(binput)
        fh[].write(PKL_BUILD)
        fh.write_pickle_boolean(false)
        fh[].write(PKL_EMPTY_LIST)
        fh.write_pickle_binput(binput)

        # now we dump objects

        if elem_count > 0:
            fh[].write(PKL_MARK)

        # fh[].write(PKL_APPENDS)

proc write_pickle_finish(fh: ptr File, binput: var uint32, elem_count: uint): void =
    if elem_count > 0:
        fh[].write(PKL_APPENDS)
    
    fh[].write(PKL_TUPLE)
    fh.write_pickle_binput(binput)
    fh[].write(PKL_BUILD)
    fh[].write(PKL_STOP)

proc write_pickle_date(fh: ptr File, value: PY_Date, binput: var uint32): void =
    fh.write_pickle_global("datetime", "date")
    fh.write_pickle_binput(binput)
    fh[].write(PKL_SHORT_BINBYTES)
    fh[].write('\4') # date has 4 bytes 2(y)-1(m)-1(d)

    var year: uint16
    year.unsafeAddr.bigEndian16(value.year.unsafeAddr)

    discard fh[].writeBuffer(year.unsafeAddr, 2)
    discard fh[].writeBuffer(value.month.unsafeAddr, 1)
    discard fh[].writeBuffer(value.day.unsafeAddr, 1)

    fh.write_pickle_binput(binput)
    fh[].write(PKL_TUPLE1)
    fh.write_pickle_binput(binput)
    fh[].write(PKL_REDUCE)
    fh.write_pickle_binput(binput)

proc write_pickle_time(fh: ptr File, value: PY_Time, binput: var uint32): void =
    fh.write_pickle_global("datetime", "time")
    fh.write_pickle_binput(binput)
    fh[].write(PKL_SHORT_BINBYTES)
    fh[].write('\6')

    var microsecond: uint32
    microsecond.unsafeAddr.bigEndian32(value.microsecond.unsafeAddr)

    var ptr_microseconds = cast[pointer](cast[int](microsecond.unsafeAddr) + 1)

    discard fh[].writeBuffer(value.hour.unsafeAddr, 1)
    discard fh[].writeBuffer(value.minute.unsafeAddr, 1)
    discard fh[].writeBuffer(value.second.unsafeAddr, 1)
    discard fh[].writeBuffer(ptr_microseconds, 3)
    fh.write_pickle_binput(binput)

    if not value.has_tz:
        fh[].write(PKL_TUPLE1)
    else:
        fh.write_pickle_global("datetime", "timezone")
        fh.write_pickle_binput(binput)
        fh.write_pickle_global("datetime", "timedelta")
        fh.write_pickle_binput(binput)
        fh.write_pickle_binint(value.tz_days)
        fh.write_pickle_binint(value.tz_seconds)
        fh.write_pickle_binint(value.tz_microseconds)
        fh[].write(PKL_TUPLE3)
        fh.write_pickle_binput(binput)
        fh[].write(PKL_REDUCE)
        fh.write_pickle_binput(binput)
        fh[].write(PKL_TUPLE1)
        fh.write_pickle_binput(binput)
        fh[].write(PKL_REDUCE)
        fh.write_pickle_binput(binput)
        fh[].write(PKL_TUPLE2)

    fh.write_pickle_binput(binput)
    fh[].write(PKL_REDUCE)
    fh.write_pickle_binput(binput)

proc write_pickle_obj[T: int|float|PY_NoneType|string|bool|PY_Date|PY_Time](fh: ptr File, value: T, binput: var uint32): void =
    when T is PY_NoneType:
        fh[].write(PKL_NONE)
        return
    when T is int:
        fh.write_pickle_binint(value)
        return
    when T is float:
        fh.write_pickle_binfloat(value)
        return
    when T is string:
        fh.write_pickle_binunicode(value)
        return
    when T is bool:
        fh.write_pickle_boolean(value)
        return
    when T is PY_Date:
        fh.write_pickle_date(value, binput)
        return
    when T is PY_Time:
        fh.write_pickle_time(value, binput)
        return
    raise newException(Exception, "not implemented error: " & $value)

proc text_reader_task(
    path: string, encoding: Encodings, dialect: Dialect, 
    destinations: var seq[string], field_relation: var OrderedTable[uint, uint], 
    row_offset: uint, row_count: int): void =
    var obj = newReaderObj(dialect)
    
    let fh = newFile(path, encoding)
    let keys_field_relation = collect: (for k in field_relation.keys: k)
    let n_columns = keys_field_relation.len()
    let guess_dtypes = true
    let n_pages = destinations.len
    
    var ranks: seq[Rank]
    
    if guess_dtypes:
        ranks = collect(newSeqOfCap(n_columns)):
            for _ in 0..n_columns-1:
                newRank()

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        let page_file_handlers = collect(newSeqOfCap(n_pages)):
            for p in destinations:
                open(p, fmWrite)

        var longest_str = newSeq[uint](n_pages)
        var column_dtypes = newSeq[char](n_pages)
        var column_nones = newSeq[bool](n_pages)
        var n_rows: uint = 0
        var binput: uint32 = 0

        for (row_idx, fields, field_count) in obj.parse_csv(fh):
            if row_count >= 0 and row_idx >= (uint row_count):
                break
                
            for idx in 0..field_count-1:
                if not ((uint idx) in keys_field_relation):
                    continue

                let fidx = field_relation[uint idx]
                let field = fields[idx]

                if not guess_dtypes:
                    longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])
                else:
                    let rank = addr ranks[fidx]
                    let (is_none, dt) = rank[].update_rank(field.unsafeAddr)

                    if dt == DataTypes.DT_STRING and not is_none:
                        longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])

                    if is_none:
                        column_nones[fidx] = true

            inc n_rows

        if not guess_dtypes:
            for idx, (fh, i) in enumerate(zip(page_file_handlers, longest_str)):
                column_dtypes[idx] = 'U'
                fh.write_numpy_header("<U" & $i, n_rows)
        else:
            for i in 0..n_pages-1:
                let fh = page_file_handlers[i]
                let rank = addr ranks[i]
                var dtype = column_dtypes[i]
                var nilish = column_nones[i]

                for it in rank[].iter():
                    let dt = it[0]
                    let count = it[1]
    
                    if count == 0:
                        break

                    if dtype == '\x00':
                        case dt:
                            of DataTypes.DT_INT: dtype = 'i'
                            of DataTypes.DT_FLOAT: dtype = 'f'
                            of DataTypes.DT_STRING: dtype = 'U'
                            of DataTypes.DT_BOOLEAN: dtype ='?'
                            else: dtype = 'O'
                        continue

                    if dtype == 'f' and dt == DataTypes.DT_INT: discard
                    elif dtype == 'i' and dt == DataTypes.DT_FLOAT: dtype = 'f'
                    else: dtype = 'O'
                
                if nilish:
                    fh.write_numpy_header("|O", n_rows)
                else:
                    case dtype:
                        of 'U': fh.write_numpy_header("<U" & $ longest_str[i], n_rows)
                        of 'i': fh.write_numpy_header("<i8", n_rows)
                        of 'f': fh.write_numpy_header("<f8", n_rows)
                        of '?': fh.write_numpy_header("|b1", n_rows)
                        of 'O': fh.write_numpy_header("|O", n_rows)
                        else: raise newException(Exception, "invalid")

                column_dtypes[i] = dtype

            for idx in 0..n_pages-1:
                let fh = page_file_handlers[idx].unsafeAddr
                let dt = column_dtypes[idx]
                let nilish = column_nones[idx]
                if dt == 'O' or nilish:
                    fh.write_pickle_start(binput, n_rows)


        fh.setFilePos(int64 row_offset, fspSet)

        for (row_idx, fields, field_count) in obj.parse_csv(fh):
            if row_count >= 0 and row_idx >= (uint row_count):
                break
                
            for idx in 0..field_count-1:
                if not ((uint idx) in keys_field_relation):
                    continue

                var str = fields[idx]
                let fidx = field_relation[uint idx]
                var fh = page_file_handlers[fidx].unsafeAddr

                if not guess_dtypes:
                    for rune in str.toRunes():
                        var ch = uint32(rune)
                        discard fh[].writeBuffer(ch.unsafeAddr, 4)

                    let dt = longest_str[fidx] - (uint str.runeLen)

                    for i in 1..dt:
                        fh[].write("\x00\x00\x00\x00")
                else:
                    let dt = column_dtypes[idx]
                    let nilish = column_nones[idx]
                    var rank = ranks[idx]

                    case dt:
                        of 'U':
                            for rune in str.toRunes():
                                var ch = uint32(rune)
                                discard fh[].writeBuffer(ch.unsafeAddr, 4)

                            let dt = longest_str[fidx] - (uint str.runeLen)

                            for i in 1..dt:
                                fh[].write("\x00\x00\x00\x00")
                        of 'i':
                            if not nilish:
                                let parsed = parseInt(str)
                                discard fh[].writeBuffer(parsed.unsafeAddr, 8)
                            else:
                                try:
                                    fh.write_pickle_obj(parseInt(str), binput)
                                except ValueError:
                                    fh.write_pickle_obj(PY_None, binput)
                        of 'f':
                            if not nilish:
                                let parsed = parseFloat(str)
                                discard fh[].writeBuffer(parsed.unsafeAddr, 8)
                            else:
                                try:
                                    fh.write_pickle_obj(parseFloat(str), binput)
                                except ValueError:
                                    fh.write_pickle_obj(PY_None, binput)
                        of '?': fh[].write((if str.toLower() == "true": '\x01' else: '\x00'))
                        of 'O': 
                            for r_addr in rank.iter():
                                let dt = r_addr[0]
                                try:
                                    case dt:
                                        of DataTypes.DT_INT:
                                            fh.write_pickle_obj(str.parse_int(), binput)
                                        of DataTypes.DT_FLOAT:
                                            fh.write_pickle_obj(str.parse_float(), binput)
                                        of DataTypes.DT_BOOLEAN:
                                            fh.write_pickle_obj(str.parse_bool(), binput)
                                        of DataTypes.DT_DATE:
                                            fh.write_pickle_obj(str.unsafeAddr.parse_date(), binput)
                                        of DataTypes.DT_TIME:
                                            fh.write_pickle_obj(str.unsafeAddr.parse_time(), binput)
                                        of DataTypes.DT_DATETIME:
                                            # discard str.parse_datetime()
                                            raise newException(Exception, "not yet implemented: " & $dt)
                                        of DataTypes.DT_STRING:
                                            if str in ["null", "Null", "NULL", "#N/A", "#n/a", "", "None"]:
                                                fh.write_pickle_obj(PY_None, binput)
                                            else:
                                                fh.write_pickle_obj(str, binput)
                                        else:
                                            raise newException(Exception, "invalid type")
                                except ValueError:
                                    continue
                                break
                        else: raise newException(Exception, "invalid")

        for idx in 0..n_pages-1:
            let fh = page_file_handlers[idx].unsafeAddr
            let dt = column_dtypes[idx]
            let nilish = column_nones[idx]
            if dt == 'O' or nilish:
                fh.write_pickle_finish(binput, n_rows)

        for f in page_file_handlers:
            f.close()

    finally:
        fh.close()

proc import_file(path: string, encoding: Encodings, dia: Dialect, columns: ptr seq[string], execute: bool, multiprocess: bool): void =
    echo "Collecting tasks: '" & path & "'"
    let (newline_offsets, newlines) = find_newlines(path, encoding)

    let dirname = "/media/ratchet/hdd/tablite/nim/page"

    if not dirExists(dirname):
        createDir(dirname)

    if newlines > 0:
        let fields = read_columns(path, encoding, dia, newline_offsets[0])

        var imp_columns: seq[string]

        if columns == nil:
            imp_columns = fields
        else:
            raise newException(Exception, "not implemented error:column selection")

        let new_fields = collect(initOrderedTable()):
            for ix, name in enumerate(fields):
                if name in imp_columns:
                    {uint ix: name}

        let inp_fields = collect(initOrderedTable()):
            for ix, name in new_fields.pairs:
                {ix: name}

        var field_relation = collect(initOrderedTable()):
            for i, c in enumerate(inp_fields.keys):
                {c: uint i}

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
            var pages = newSeq[string](fields.len)

            for idx in 0..fields.len - 1:
                pages[idx] = dirname & "/" & $page_idx & ".npy"
                inc page_idx

            if not multiprocess:
                text_reader_task(path, encoding, dia, pages, field_relation, newline_offsets[row_idx], int page_size)

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
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", ENC_UTF8)
        (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", ENC_UTF16)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", ENC_UTF16)

        let d0 = getTime()
        import_file(path_csv, encoding, dialect, nil, true, false)
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
                    {parseUInt(k): parseUInt(v)}

            let offset = parseUInt(opts.task.get.offset)
            let count = parseInt(opts.task.get.count)

            text_reader_task(path, encoding, dialect, pages, field_relation, offset, count)