# start with standard ubuntu:22.04
# TODO: use multiple stage build for libdds.so & python dependance
FROM docker.io/ubuntu:jammy

# libdds-dev contains libdds.so
RUN apt-get update && \
    apt-get -y install libboost-thread-dev libdds-dev python3.10 python-is-python3 python3-pip  && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean

# https://linuxhint.com/install-use-mono-ubuntu-22-04/

# mono is needed for EPBot, skipped in container
#RUN echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list 
#RUN apt-get update && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends mono-complete 

ADD requirements.txt /app/

WORKDIR /app
RUN pip install -r requirements.txt

ADD src /app/src
ADD models /app/models
ADD start_ben_all.sh /app/

EXPOSE 4443 8080
CMD ./start_ben_all.sh
