import std/[math, sets, sugar, tables, enumerate]
import strformat
import nimpy as nimpy
import ../[pymodules, numpy, pytypes, nimpyext]
from ../utils import uniqueName, implement

proc collectColumnNames*(table: nimpy.PyObject): seq[string] =
    var columns: seq[string] = collect:
        for k in table.columns.keys():
            k.to(string)
    return columns

proc accumulate[T: int64 | float64](page: var Int8NDArray | var Int16NDArray | var Int32NDArray | var Int64NDArray, buffer: var seq[T], bufSize: var int, value: var float): void = # var uses reference
    bufSize = page.len
    for (i, rowValue) in enumerate(page.pgIter):
        value += float64 rowValue
        when T is float64:
            buffer[i] = value
        else:
            buffer[i] = int64 value

proc accumulate(page: var Float32NDArray | var Float64NDArray, buffer: var seq[float64], bufSize: var int, value: var float): void = # var uses reference
    bufSize = page.len
    for (i, rowValue) in enumerate(page.pgIter):
        value += rowValue
        buffer[i] = value

proc accumulate[T: int64 | float64](page: var ObjectNDArray, buffer: var seq[T], bufSize: var int, value: var float): void = # var uses reference
    
    template setValue() =
        when T is float64:
            buffer[i] = value
        else:
            buffer[i] = int64 value
    
    bufSize = page.len
    for (i, rowValue) in enumerate(page.pgIter):
        case rowValue.kind:
        of K_INT:
            value += float64 PY_Int(rowValue).value
            setValue()
        of K_FLOAT:
            value += PY_Float(rowValue).value
            when T is float64:
                buffer[i] = value
        else:
            setValue()

proc accumulate*(table: nimpy.PyObject, column: string, result_name: string = "accumulate", result_only: bool = false, tqdm: nimpy.PyObject = pymodules.tqdm().tqdm): (nimpy.PyObject, nimpy.PyObject) {.exportpy.} =
    if isNone(table):
        raise newException(Exception, "missing table")
    
    var columns: seq[string] = collectColumnNames(table)

    if not columns.contains(column):
        raise newException(Exception, "column not found")
    
    var pagePaths = collect:
        for p in table[column].pages:
            builtins().str(p.path).to(string)
    var 
        TableType: nimpy.PyObject = builtins().type(table)
        resName: string = uniqueName(result_name, columns)
        tabliteConf = tabliteConfig().Config
        columnTypes = getColumnTypes(pagePaths)
        bufFloat {.noinit.}: seq[float64]
        bufInt {.noinit.}: seq[int64]
        isFloat: bool = false
        pageSize: int = tabliteConf.PAGE_SIZE.to(int)
        pid: string = tabliteConf.pid.to(string)
        workDir: string = builtins().str(tabliteConf.workdir).to(string)
        pidDir: string = &"{workDir}/{pid}"
        bufSize: int = 0
        accumulatedValue: float = 0.0
        newPages: seq[nimpy.PyObject]
        pbarSteps: int = pagePaths.len
    var pbar = tqdm!(total: pbarSteps, desc: "reducing with 'accumulate'")

    if K_FLOAT in columnTypes:
        bufFloat = newSeq[float64](pageSize)
        bufInt = newSeq[int64](0)
        isFloat = true
    elif K_INT in columnTypes:
        bufFloat = newSeq[float64](0)
        bufInt = newSeq[int64](pageSize)
    else:
        bufFloat = newSeq[float64](0)
        bufInt = newSeq[int64](pageSize)

    template callAccumulator(T: typedesc) =
        when T is Float32NDArray or T is Float64NDArray:
            T(page).accumulate(bufFloat, bufSize, accumulatedValue)
        else:
            if isFloat:
                T(page).accumulate(bufFloat, bufSize, accumulatedValue)
            else:
                T(page).accumulate(bufInt, bufSize, accumulatedValue)

    for pagePath in pagePaths:
        var page: BaseNDArray = readNumpy(pagePath)
        case page.kind:
        of K_INT8: Int8NDArray.callAccumulator()
        of K_INT16: Int16NDArray.callAccumulator()
        of K_INT32: Int32NDArray.callAccumulator()
        of K_INT64: Int64NDArray.callAccumulator()
        of K_FLOAT32: Float32NDArray.callAccumulator()
        of K_FLOAT64: Float64NDArray.callAccumulator()
        of K_OBJECT: ObjectNDArray.callAccumulator()
        else:
            bufSize = page.len

        var 
            newPageId: string = tabliteBase().SimplePage.next_id(pidDir).to(string)
            newPage {.noinit.}: BaseNDArray
            dtypes = initTable[KindObjectND, int]()
            shape: seq[int] = @[bufSize]

        if isFloat:
            newPage = Float64NDArray(buf: bufFloat, shape: shape, kind: K_FLOAT64)
            dtypes[K_FLOAT] = bufSize
        else:
            newPage = Int64NDArray(buf: bufInt, shape: shape, kind: K_INT64)
            dtypes[K_INT] = bufSize
        
        newPage.save(&"{pidDir}/pages/{newPageId}.npy")
        newPages.add(newPyPage(newPageId, pidDir, bufSize, dtypes))
        discard pbar.update(1)
    
    var newColumn = tabliteBase().Column(pidDir)
    newColumn.pages = newPages

    var tblFail = TableType!(initTable[nimpy.PyObject, nimpy.PyObject]())

    if result_only:
        var tblPass = TableType!({ result_name: newColumn }.toTable())
        return (tblPass, tblFail)

    var tblPass = table.copy()
    tblPass[resName] = newColumn

    return (tblPass, tblFail)