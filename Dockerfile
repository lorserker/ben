# start with standard ubuntu:24.04 (noble) which ships Python 3.12
# TODO: use multiple stage build for libdds.so & python dependance
FROM docker.io/ubuntu:noble

RUN apt-get update && \
    apt-get -y install python3.12 python-is-python3 python3-pip python3-gdbm && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    apt-get clean

# .NET 10 runtime for ACE (CoreCLR) — libicu is required for .NET globalization
RUN apt-get -y install curl libicu74 && \
    curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 10.0 --runtime dotnet --install-dir /usr/share/dotnet && \
    apt-get clean
ENV DOTNET_ROOT=/usr/share/dotnet

ADD requirements.txt /app/

WORKDIR /app
# noble enforces PEP 668 (externally-managed) - the container is the isolated env
RUN pip install --break-system-packages -r requirements.txt

# Suppress TensorFlow/CUDA warnings (no GPU in container)
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

COPY src/frontend /app/frontend/
COPY src/*.py /app/
ADD src/bidding /app/bidding/
ADD src/nn /app/nn/
ADD src/ddsolver /app/ddsolver/
ADD src/alphamju /app/alphamju/
ADD src/ace /app/ace/
ADD src/bba /app/bba/
ADD src/pimc /app/pimc/
ADD src/suitc /app/suitc/
ADD src/openinglead /app/openinglead/
# bin/ includes the DDS 3.0.0 solver extension under bin/dds3-linux/.
# It MUST be built for this image's Python (3.12): build from the DDS repo
# (bazel build //python:_dds3 for 3.12) and vendor the dds3 package there before
# building this image. A 3.10 _dds3.so will fail to import. See src/ddsolver/ddsolver.py.
ADD bin /app/bin/
ADD src/config /app/config/
ADD models /app/models/
COPY "BBA/CC/" "/app/BBA/CC/"
ADD start_ben_all.sh /app/

# BBA imports from "src.objects" — create symlink so /app/src -> /app
# Grant execution permissions to the script and fix line endings
RUN ln -s /app /app/src && \
    ln -s /app/BBA /BBA && \
    sed -i 's/\r$//' /app/start_ben_all.sh && chmod +x /app/start_ben_all.sh

EXPOSE 8080 4443 8085
CMD ./start_ben_all.sh
