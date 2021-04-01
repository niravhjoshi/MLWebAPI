FROM python:3.8

WORKDIR /FlaskWebMLapi

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt


EXPOSE 5005

COPY . /FlaskWebMLapi

CMD ["python3", "./app.py"]