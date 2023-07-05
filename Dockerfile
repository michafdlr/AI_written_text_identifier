FROM python:3.10.6-buster

WORKDIR /prod

RUN pip install --upgrade pip

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY ai_written_text_identifier ai_written_text_identifier
COPY setup.py  setup.py
COPY models models
COPY tokenizers tokenizers
COPY extractors extractors
RUN pip install .

CMD uvicorn ai_written_text_identifier.api.fast:app --host 0.0.0.0 --port $PORT
