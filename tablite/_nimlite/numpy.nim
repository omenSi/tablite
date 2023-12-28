import std/[unicode, strutils, sugar, times]
import dateutils, pytypes, unpickling, utils
import pymodules as pymodules
import nimpy as nimpy, nimpy/raw_buffers

const NUMPY_MAGIC = "\x93NUMPY"
const NUMPY_MAJOR = "\x01"
const NUMPY_MINOR = "\x00"
const NUMPY_MAGIC_LEN = NUMPY_MAGIC.len
const NUMPY_MAJOR_LEN = NUMPY_MAJOR.len
const NUMPY_MINOR_LEN = NUMPY_MINOR.len

type NDArrayTypeDescriptor = enum
    D_OBJECT
    D_UNICODE
    D_BOOLEAN
    D_INT
    D_FLOAT
    D_TIME
    D_DATE_DAYS
    D_DATETIME_SECONDS
    D_DATETIME_MILISECONDS
    D_DATETIME_MICROSECONDS

type NDArrayDescriptor = (Endianness, NDArrayTypeDescriptor, int)
type BaseNDArray* = ref object of RootObj
    shape*: Shape

type BooleanNDArray* = ref object of BaseNDArray
    buf*: seq[bool]

type Int8NDArray* = ref object of BaseNDArray
    buf*: seq[int8]
type Int16NDArray* = ref object of BaseNDArray
    buf*: seq[int16]
type Int32NDArray* = ref object of BaseNDArray
    buf*: seq[int32]
type Int64NDArray* = ref object of BaseNDArray
    buf*: seq[int64]

type Float32NDArray* = ref object of BaseNDArray
    buf*: seq[float32]
type Float64NDArray* = ref object of BaseNDArray
    buf*: seq[float64]

type DateNDArray* = ref object of BaseNDArray
    buf*: seq[DateTime]

type DateTimeNDArray* = ref object of BaseNDArray
    buf*: seq[DateTime]

type UnicodeNDArray* = ref object of BaseNDArray
    buf*: seq[char]
    size*: int

type ObjectNDArray* = ref object of BaseNDArray
    buf*: seq[PyObjectND]

proc writeNumpyHeader*(fh: File, dtype: string, shape: uint): void =
    let header = "{'descr': '" & dtype & "', 'fortran_order': False, 'shape': (" & $shape & ",)}"
    let header_len = len(header)
    let padding = (64 - ((NUMPY_MAGIC_LEN + NUMPY_MAJOR_LEN + NUMPY_MINOR_LEN + 2 + header_len)) mod 64)

    let padding_header = uint16 (padding + header_len)

    fh.write(NUMPY_MAGIC)
    fh.write(NUMPY_MAJOR)
    fh.write(NUMPY_MINOR)

    discard fh.writeBuffer(padding_header.unsafeAddr, 2)

    fh.write(header)

    for i in 0..padding-2:
        fh.write(" ")
    fh.write("\n")

proc writeNumpyUnicode*(fh: var File, str: var string, unicode_len: uint): void {.inline.} =
    for rune in str.toRunes():
        var ch = uint32(rune)
        discard fh.writeBuffer(ch.unsafeAddr, 4)

    let dt = unicode_len - (uint str.runeLen)

    for i in 1..dt:
        fh.write("\x00\x00\x00\x00")

proc writeNumpyInt*(fh: var File, value: int): void {.inline.} =
    discard fh.writeBuffer(value.unsafeAddr, 8)

proc writeNumpyFloat*(fh: var File, value: float): void {.inline.} =
    discard fh.writeBuffer(value.unsafeAddr, 8)

proc writeNumpyBool*(fh: var File, str: var string): void {.inline.} =
    fh.write((if str.toLower() == "true": '\x01' else: '\x00'))


proc repr(self: ObjectNDArray): string =
    let elems = collect: (for e in self.buf: $e)
    return "ObjectNDArray(buf: @[" & elems.join(", ") & "], shape: " & $self.shape & ")"

proc `$`*(self: BaseNDArray): string =
    if self of BooleanNDArray: return repr(BooleanNDArray self)
    if self of Int64NDArray: return repr(Int64NDArray self)
    if self of Float64NDArray: return repr(Float64NDArray self)
    if self of UnicodeNDArray: return repr(UnicodeNDArray self)
    if self of ObjectNDArray: return repr(ObjectNDArray self)
    if self of DateNDArray: return repr(DateNDArray self)
    if self of DateTimeNDArray: return repr(DateTimeNDArray self)
    else: implement("BaseNDArray.`$`")


proc validateHeader(fh: File, buf: var array[NUMPY_MAGIC_LEN, uint8], header: string, header_len: int): void {.inline.} =
    if fh.readBytes(buf, 0, header_len) != header_len:
        corrupted()

    for idx in 0..header_len-1:
        if buf[idx] != uint8 header[idx]:
            corrupted()

proc consumeHeaderString(header: var string, header_len: int, offset: var int): string =
    var start_index = -1
    var end_index = -1
    var is_inside = false

    while offset < header_len:
        if header[offset] == '\'':
            if is_inside:
                end_index = offset
                is_inside = false
                break
            else:
                start_index = offset
                is_inside = true

        offset = offset + 1

    if start_index == end_index or start_index < 0 or end_index < 0:
        corrupted()

    offset = offset + 1

    return header[start_index+1..end_index-1]

proc consumeBool(header: var string, header_len: int, offset: var int): bool =
    while offset < header_len:
        if header[offset].isAlphaNumeric():
            break

        offset = offset + 1

    var start_index = offset
    var end_index = -1

    while offset < header_len:
        if header[offset] == ',':
            end_index = offset - 1
            break

        offset = offset + 1

    var res_str = $header[start_index..end_index]
    var res: bool

    case res_str.toLower():
        of "true": res = true
        of "false": res = false
        else: corrupted()

    offset = offset + 1

    return res

proc consumeShape(header: var string, header_len: int, offset: var int): Shape =
    var shape = newSeq[int]()
    var start_index = -1
    var end_index = -1

    while offset < header_len:
        if header[offset] == ')':
            end_index = offset - 1
            break

        if header[offset] == '(':
            start_index = offset + 1

        offset = offset + 1

    if start_index == end_index or start_index < 0 or end_index < 0: corrupted()

    let shape_str_seq = header[start_index..end_index].split(',')
    let len_shape_str_seq = shape_str_seq.len

    for i in 0..len_shape_str_seq-1:
        let sh_str = shape_str_seq[i]

        if sh_str.len == 0:
            if i + 1 != len_shape_str_seq: corrupted()
            continue

        shape.add(parseInt(sh_str))

    return shape

proc consumeDescr(header: var string, header_len: int, offset: var int): NDArrayDescriptor =
    let descr = consumeHeaderString(header, header_len, offset)

    while offset < header_len:
        if header[offset] == ',':
            break

        offset = offset + 1

    offset = offset + 1

    var endianness: Endianness
    let valid_types = ['b', 'i', 'f', 'U', 'O', 'm', 'M']

    var type_offset: int = 1

    case descr[0]:
        of '|', '<': endianness = Endianness.littleEndian
        of '>':
            endianness = Endianness.bigEndian
        else:
            if not (descr[0] in valid_types): corrupted()

            type_offset = 0

    let type_string = descr[type_offset]

    if not (type_string in valid_types): corrupted()

    var size: int
    var descriptor: NDArrayTypeDescriptor
    var dt_descriptor: string

    if type_string == 'm' or type_string == 'M':
        if descr[type_offset + 1] != '8' or descr[type_offset + 2] != '[' or descr[^1] != ']':
            corrupted()

        dt_descriptor = descr[(type_offset + 3)..^2]

    case type_string:
        of 'O':
            if (type_offset + 1) != descr.len: corrupted()
            size = -1
            descriptor = NDArrayTypeDescriptor.D_OBJECT
        of 'm':
            case dt_descriptor:
            else: implement(descr)
        of 'M':
            case dt_descriptor:
            of "D":
                size = 8
                descriptor = NDArrayTypeDescriptor.D_DATE_DAYS
            of "us":
                size = 8
                descriptor = NDArrayTypeDescriptor.D_DATETIME_MICROSECONDS
            else: implement(descr)
        else:
            size = parseInt(descr[type_offset+1..descr.len-1])

            case type_string:
                of 'b':
                    if size != 1: corrupted()
                    descriptor = NDArrayTypeDescriptor.D_BOOLEAN
                of 'i':
                    if size != 8: implement("int size != 8")
                    descriptor = NDArrayTypeDescriptor.D_INT
                of 'f':
                    if size != 8: implement("float size != 8")
                    descriptor = NDArrayTypeDescriptor.D_FLOAT
                of 'U':
                    if size <= 0: corrupted()
                    descriptor = NDArrayTypeDescriptor.D_UNICODE
                else:
                    # never happens
                    corrupted()


    return (endianness, descriptor, size)

proc parseHeader(header: var string): (NDArrayDescriptor, bool, Shape) =
    var offset = 0
    var header_len = header.len
    var entry_consumed = false

    while offset < header_len:
        # consume entry token
        if header[offset] == '{':
            entry_consumed = true
            break

        offset = offset + 1

    if not entry_consumed:
        corrupted()

    offset = offset + 1

    var descr: NDArrayDescriptor
    var has_descr = false
    var order: bool
    var has_order = false
    var shape: Shape
    var has_shape = false

    for i in 0..2:
        let name = consumeHeaderString(header, header_len, offset)

        case name:
            of "descr":
                descr = consumeDescr(header, header_len, offset)
                has_descr = true
            of "fortran_order":
                order = consumeBool(header, header_len, offset)
                has_order = true
            of "shape":
                shape = consumeShape(header, header_len, offset)
                has_shape = true
            else:
                corrupted()

    if not has_descr or not has_order or not has_shape:
        corrupted()

    return (descr, order, shape)



template readPrimitiveBuffer[T: typed](fh: var File, shape: var Shape): seq[T] =
    var elements = calcShapeElements(shape)
    var buf {.noinit.} = newSeq[T](elements)
    var size_T = sizeof(T)
    var buffer_size = elements * size_T

    if fh.readBuffer(addr buf[0], buffer_size) != buffer_size:
        corrupted()

    buf

proc newBooleanNDArray(fh: var File, shape: var Shape): BooleanNDArray =
    return BooleanNDArray(
        buf: readPrimitiveBuffer[bool](fh, shape),
        shape: shape
    )

template newIntNDArray(fh: var File, endianness: Endianness, size: int, shape: var Shape) =
    case size:
        of 1: Int8NDArray(buf: readPrimitiveBuffer[int8](fh, shape), shape: shape)
        of 2: Int16NDArray(buf: readPrimitiveBuffer[int16](fh, shape), shape: shape)
        of 4: Int32NDArray(buf: readPrimitiveBuffer[int32](fh, shape), shape: shape)
        of 8: Int64NDArray(buf: readPrimitiveBuffer[int64](fh, shape), shape: shape)
        else: corrupted()

proc newDateArray_Days(fh: var File, endianness: Endianness, shape: var Shape): DateNDArray {.inline.} =
    let buf = collect: (for v in readPrimitiveBuffer[int64](fh, shape): days2Date(v))

    return DateNDArray(buf: buf, shape: shape)

proc newDateTimeArray_Seconds(fh: var File, endianness: Endianness, shape: var Shape): DateTimeNDArray {.inline.} =
    let data = readPrimitiveBuffer[int64](fh, shape)
    let buf = collect: (for v in data: initTime(v, 0).utc())

    return DateTimeNDArray(buf: buf, shape: shape)

proc newDateTimeArray_Miliseconds(fh: var File, endianness: Endianness, shape: var Shape): DateTimeNDArray {.inline.} =
    let data = readPrimitiveBuffer[int64](fh, shape)
    let buf = collect:
        for v in data:
            let (s, m) = divmod(v, 1000)
            initTime(s, m * 1000).utc()

    return DateTimeNDArray(buf: buf, shape: shape)

proc newDateTimeArray_Microseconds(fh: var File, endianness: Endianness, shape: var Shape): DateTimeNDArray {.inline.} =
    let data = readPrimitiveBuffer[int64](fh, shape)
    let buf = collect:
        for v in data:
            let (s, u) = divmod(v, 1_000_000)
            initTime(s, u).utc()

    return DateTimeNDArray(buf: buf, shape: shape)

template newFloatNDArray(fh: var File, endianness: Endianness, size: int, shape: var Shape) =
    case size:
        of 4: Float32NDArray(buf: readPrimitiveBuffer[float32](fh, shape), shape: shape)
        of 8: Float64NDArray(buf: readPrimitiveBuffer[float64](fh, shape), shape: shape)
        else: corrupted()

proc newUnicodeNDArray(fh: var File, endianness: Endianness, size: int, shape: var Shape): UnicodeNDArray =
    var elements = calcShapeElements(shape)
    var elem_size = elements * size * 4
    var buf {.noinit.} = newSeq[char](elem_size)

    if fh.readBuffer(addr buf[0], elem_size) != elem_size:
        corrupted()

    return UnicodeNDArray(buf: buf, shape: shape, size: size)

proc newObjectNDArray(fh: var File, endianness: Endianness, shape: var Shape): ObjectNDArray =
    var elements = calcShapeElements(shape)
    var (shape, buf) = readPickledPage(fh, endianness, shape)

    if calcShapeElements(shape) != elements:
        corrupted()

    return ObjectNDArray(shape: shape, buf: buf)

proc readPageInfo(fh: var File): (NDArrayDescriptor, bool, Shape) =
    var header_bytes: array[NUMPY_MAGIC_LEN, uint8]

    validateHeader(fh, header_bytes, NUMPY_MAGIC, NUMPY_MAGIC_LEN)
    validateHeader(fh, header_bytes, NUMPY_MAJOR, NUMPY_MAJOR_LEN)
    validateHeader(fh, header_bytes, NUMPY_MINOR, NUMPY_MINOR_LEN)

    var header_size: uint16

    if fh.readBuffer(addr header_size, 2) != 2:
        corrupted()

    var header = newString(header_size)

    if fh.readBuffer(addr header[0], int header_size) != int header_size:
        corrupted()

    var (descr, order, shape) = parseHeader(header)

    if order: implement("fortran_order")
    if shape.len != 1: implement("shape.len != 1")

    return (descr, order, shape)

proc readNumpy(fh: var File): BaseNDArray =
    var ((descr_endianness, descr_type, descr_size), _, shape) = readPageInfo(fh)
    var page: BaseNDArray

    case descr_type:
        of D_BOOLEAN: page = newBooleanNDArray(fh, shape)
        of D_INT: page = newIntNDArray(fh, descr_endianness, descr_size, shape)
        of D_FLOAT: page = newFloatNDArray(fh, descr_endianness, descr_size, shape)
        of D_UNICODE: page = newUnicodeNDArray(fh, descr_endianness, descr_size, shape)
        of D_OBJECT: page = newObjectNDArray(fh, descr_endianness, shape)
        of D_DATE_DAYS: page = newDateArray_Days(fh, descr_endianness, shape)
        of D_DATETIME_SECONDS: page = newDateTimeArray_Seconds(fh, descr_endianness, shape)
        of D_DATETIME_MILISECONDS: page = newDateTimeArray_Miliseconds(fh, descr_endianness, shape)
        of D_DATETIME_MICROSECONDS: page = newDateTimeArray_Microseconds(fh, descr_endianness, shape)
        else: implement($descr_type)

    return page

proc readNumpy*(path: string): BaseNDArray =
    var fh = open(path, fmRead)
    let arr = readNumpy(fh)

    fh.close()

    return arr

proc toNumpyPrimitive(arrType: string, shape: var Shape, sizeof: int, buf: pointer): nimpy.PyObject =
    let elements = calcShapeElements(shape)
    let np = pyImport("numpy")
    var ndBuf: RawPyBuffer
    let ndArray = np.ndarray(shape, arrType)

    ndArray.getBuffer(ndBuf, PyBUF_WRITABLE or PyBUF_ND)

    ndBuf.buf.copyMem(buf, sizeof * elements)
    ndBuf.release()

    return ndArray

proc toNumpyPrimitive[T: bool | int8 | int16 | int32 | int64 | float32 | float64](shape: var Shape, buf: pointer): nimpy.PyObject =
    when T is bool:
        return toNumpyPrimitive("?", shape, sizeof(T), buf)
    else:
        let sz = sizeof(T)

        when T is int8 or T is int16 or T is int32 or T is int64:
            return toNumpyPrimitive("i" & $sz, shape, sz, buf)
        else:
            when T is float32 or T is float64:
                return toNumpyPrimitive("f" & $sz, shape, sz, buf)
            else:
                corrupted()

proc toPython(self: BooleanNDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[bool](self.shape, addr self.buf[0])

proc toPython(self: Int8NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int8](self.shape, addr self.buf[0])
proc toPython(self: Int16NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int16](self.shape, addr self.buf[0])
proc toPython(self: Int32NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int32](self.shape, addr self.buf[0])
proc toPython(self: Int64NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int64](self.shape, addr self.buf[0])

proc toPython(self: Float32NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[float32](self.shape, addr self.buf[0])
proc toPython(self: Float64NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[float64](self.shape, addr self.buf[0])

proc toPython(self: UnicodeNDArray): nimpy.PyObject = toNumpyPrimitive("U" & $self.size, self.shape, self.size * 4, addr self.buf[0])

proc toPython(self: DateNDArray): nimpy.PyObject =
    var buf = collect:
        for el in self.buf:
            el.toTime.time2Duration.inDays

    return toNumpyPrimitive("M8[D]", self.shape, sizeof(int64), addr buf[0])

proc toPython(self: DateTimeNDArray): nimpy.PyObject =
    var buf = collect:
        for el in self.buf:
            el.toTime.time2Duration.inMicroseconds

    return toNumpyPrimitive("M8[us]", self.shape, sizeof(int64), addr buf[0])

proc toNimpy(self: PY_NoneType): nimpy.PyObject = pymodules.builtins().None
proc toNimpy(self: PY_Boolean): nimpy.PyObject = pymodules.builtins().bool(self.value)
proc toNimpy(self: PY_Int): nimpy.PyObject = pymodules.builtins().int(self.value)
proc toNimpy(self: PY_Float): nimpy.PyObject = pymodules.builtins().float(self.value)
proc toNimpy(self: PY_String): nimpy.PyObject = pymodules.builtins().str(self.value)
proc toNimpy(self: PY_Date): nimpy.PyObject =
    return pymodules.datetime().date(self.value.year, self.value.month, self.value.monthday)
proc toNimpy(self: PY_Time): nimpy.PyObject =
    let hour = self.getHour()
    let minute = self.getMinute()
    let second = self.getSecond()
    let microsecond = self.getMicrosecond()

    return pymodules.datetime().time(
        hour = hour, minute = minute, second = second, microsecond = microsecond
    )

proc toNimpy(self: PY_DateTime): nimpy.PyObject =
    return pymodules.datetime().datetime(
        self.value.year, self.value.month, self.value.monthday,
        self.value.hour, self.value.minute, self.value.second, int(self.value.nanosecond / 1000)
    )

proc toNimpy(self: PY_ObjectND): nimpy.PyObject =
    if self of PY_NoneType: return PY_NoneType(self).toNimpy()
    if self of PY_Boolean: return PY_Boolean(self).toNimpy()
    if self of PY_Int: return PY_Int(self).toNimpy()
    if self of PY_Float: return PY_Float(self).toNimpy()
    if self of PY_String: return PY_String(self).toNimpy()
    if self of PY_Date: return PY_Date(self).toNimpy()
    if self of PY_Time: return PY_Time(self).toNimpy()
    if self of PY_DateTime: return PY_DateTime(self).toNimpy()

    implement(repr(self))

proc toPython(self: ObjectNDArray): nimpy.PyObject =
    let buf = collect:
        for el in self.buf:
            el.toNimpy()

    return numpy().array(buf)


proc toPython*(self: BaseNDArray): nimpy.PyObject =
    if self of BooleanNDArray: return BooleanNDArray(self).toPython()
    if self of Int8NDArray: return Int8NDArray(self).toPython()
    if self of Int16NDArray: return Int16NDArray(self).toPython()
    if self of Int32NDArray: return Int32NDArray(self).toPython()
    if self of Int64NDArray: return Int64NDArray(self).toPython()
    if self of Float32NDArray: return Float32NDArray(self).toPython()
    if self of Float64NDArray: return Float64NDArray(self).toPython()
    if self of DateNDArray: return DateNDArray(self).toPython()
    if self of DateTimeNDArray: return DateTimeNDArray(self).toPython()
    if self of UnicodeNDArray: return UnicodeNDArray(self).toPython()
    if self of ObjectNDArray: return ObjectNDArray(self).toPython()

    corrupted()

proc getPageLen*(fh: var File): int =
    var (_, _, shape) = readPageInfo(fh)

    return calcShapeElements(shape)

proc getPageLen*(path: string): int =
    var fh = open(path, fmRead)
    let len = getPageLen(fh)

    fh.close()

    return len

proc getColumnLen*(pages: openArray[string]): int =
    var acc = 0

    for p in pages:
        acc = acc + getPageLen(p)

    return acc

when isMainModule and appType != "lib":
    var arr = readNumpy("/home/ratchet/Documents/dematic/tablite/tests/data/pages/mixed.npy")

    echo arr

    let pyObj = pymodules.builtins().repr(arr.toPython())

    echo $pyObj
