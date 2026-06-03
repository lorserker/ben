Place the macOS DDS 3.0.0 solver extension here as:

    bin/dds3-darwin/dds3/__init__.py
    bin/dds3-darwin/dds3/_dds3.so

Build it from the DDS repository (`bazelisk build -c opt //python:_dds3`).
The __init__.py is python/dds3/__init__.py in the DDS repo; _dds3.so is the
built bazel-bin/python/_dds3.so.

Alternatively, `pip install` the dds3 wheel into BEN's Python environment and
this folder is not needed. See src/ddsolver/README.md.
