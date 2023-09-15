FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt
RUN  pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install ez_setup
RUN pip -q install git+https://github.com/huggingface/transformers
# Downgrade protobuf before installing other requirements
RUN pip install 'protobuf==3.20.*'
RUN pip install -r requirements.txt

COPY . .


ENV FLASK_APP=app
ENV FLASK_ENV=development

CMD ["flask", "run", "--host=0.0.0.0"]
