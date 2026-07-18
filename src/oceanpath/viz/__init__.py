"""Presentation-only helpers (matplotlib rendering) for OceanPath artifacts.

This package owns diagnostic *rendering*. Domain packages emit typed data (for
example, the coverage grids produced during an mmap build); ``viz`` turns that
data into images. Keeping matplotlib here means the storage/data-contract code
never imports a plotting library.
"""
