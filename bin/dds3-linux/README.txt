Place the Linux DDS 3.0.0 solver extension here as:

    bin/dds3-linux/dds3/__init__.py
    bin/dds3-linux/dds3/_dds3.so

Build it from the DDS repository (`bazel build -c opt //python:_dds3`).
The __init__.py is python/dds3/__init__.py in the DDS repo; _dds3.so is the
built bazel-bin/python/_dds3.so.

The Docker images copy this folder, so the .so must be a Linux x86_64 build
matching the container's Python (3.10). See src/ddsolver/README.md.
