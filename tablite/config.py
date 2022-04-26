import pathlib, os

class HDF5_Config(object):
    
    # The default location for the storage
    H5_STORAGE = pathlib.Path(os.getcwd()) / "tablite.hdf5"
    # to overwrite first import the config class:
    # >>> from tablite.config import Config
    # >>> Config.H5_STORAGE = /this/new/location
    # then import the Table class 
    # >>> from tablite import Table
    # for every new table or record this path will be used.

    H5_PAGE_SIZE = 1_000_000  # sets the page size limit.

    H5_ENCODING = 'UTF-8'  # sets the page encoding when using bytes



