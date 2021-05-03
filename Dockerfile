FROM python:3.6-buster

RUN apt-get update -y \
    && apt-get install -y python3-dev python3-pip build-essential \
    && apt-get install gcc -y \
    && apt-get clean

COPY requirements.txt /requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN mkdir -p /app
WORKDIR /app

COPY /src /app

CMD ["python", "run.py"]
