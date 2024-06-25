FROM python:3.9
WORKDIR /Food-Classifier

COPY ./requirements.txt /Food-Classifier/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /Food-Classifier/requirements.txt
COPY ./app /Food-Classifier/app

CMD ["fastapi", "run", "my_api.py", "--port", "80"]

