FROM python:3.7.13-bullseye

RUN pip install numpy
RUN pip install scipy
RUN pip install tensorflow==1.15
RUN pip install grpcio-tools
RUN pip install bottle
RUN pip install gevent
RUN pip install protobuf==3.20.*

RUN apt-get update && apt-get install -y libdds-dev

WORKDIR /app

COPY src ./src
COPY models ./models

WORKDIR /app/src

EXPOSE 8080

ENV PYTHONUNBUFFERED True
ENV DDS_PATH libdds.so

CMD ["python", "apiserver.py"]