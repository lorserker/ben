# start with standard ubuntu:22.04
# TODO: use multiple stage build for libdds.so & python dependance
FROM ubuntu:jammy

RUN apt-get update && \
    apt-get -y install libboost-thread-dev python3.10 python-is-python3 python3-pip  && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean

ADD .

RUN pip install -r requirements.txt

# CMD