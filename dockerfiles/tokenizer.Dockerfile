FROM python:3.10-alpine3.16

RUN pip install datasets transformers mwparserfromhell apache-beam ftfy

COPY train_tokenizer.py /train_tokenizer.py