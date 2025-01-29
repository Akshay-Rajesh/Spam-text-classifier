FROM python:3.11-bullseye
VOLUME /content
COPY  . "/content"
RUN pip install -r /content/requirements.txt
WORKDIR /content
CMD cd /content
CMD python predict.py
EXPOSE 80

