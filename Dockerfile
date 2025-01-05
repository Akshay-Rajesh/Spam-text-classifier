FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN cat requirements.txt | grep -v 'pywin32==' | pip install --no-cache-dir -r /dev/stdin

EXPOSE 80

ENV NAME World

CMD ["python", "predict.py"]
