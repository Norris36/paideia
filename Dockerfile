# syntax=docker/dockerfile:1

FROM python:3.11.2-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

## can you set the port here, so we switch from 8501 to 8000?

EXPOSE 8501

CMD [ "streamlit", "run", "paideia.py"]

