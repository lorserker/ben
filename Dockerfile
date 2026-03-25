# start with standard ubuntu:22.04
# TODO: use multiple stage build for libdds.so & python dependance
FROM docker.io/ubuntu:jammy

# libdds-dev contains libdds.so
RUN apt-get update && \
    apt-get -y install libboost-thread-dev libdds-dev python3.10 python-is-python3 python3-pip  && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean

# .NET 10 runtime for EPBot (CoreCLR) — libicu is required for .NET globalization
RUN apt-get -y install curl libicu70 && \
    curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 10.0 --runtime dotnet --install-dir /usr/share/dotnet && \
    apt-get clean
ENV DOTNET_ROOT=/usr/share/dotnet

ADD requirements.txt /app/

WORKDIR /app
RUN pip install -r requirements.txt

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
