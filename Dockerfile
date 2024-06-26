FROM python:3.9
WORKDIR /Food-Classifier

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY ./app ./app


CMD ["fastapi", "run", "./app/my_api.py", "--port", "80"]

