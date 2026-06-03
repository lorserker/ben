Place the Windows DDS 3.0.0 solver extension here as:

    bin/dds3-win/dds3/__init__.py
    bin/dds3-win/dds3/_dds3.pyd

Build it from the DDS repository (run build_dds3.cmd, or
`bazel build -c opt //python:_dds3`). The __init__.py is python/dds3/__init__.py
in the DDS repo; _dds3.pyd is the built bazel-bin/python/_dds3.pyd.

Alternatively, `pip install` the dds3 wheel into BEN's Python environment and
this folder is not needed. See src/ddsolver/README.md.
