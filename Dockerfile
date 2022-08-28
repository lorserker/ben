FROM python:3.7-buster

RUN pip install numpy
RUN pip install scipy
RUN pip install tensorflow==1.15
RUN pip install grpcio-tools
RUN pip install bottle
RUN pip install gevent

WORKDIR /app

COPY src ./src
COPY models ./models

WORKDIR /app/src

EXPOSE 8080

CMD ["python", "apiserver.py"]