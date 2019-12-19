FROM python:3.6-slim-buster
LABEL maintainer="akaash - find me on linkedin/akaashsidhu"

RUN apt-get update && apt-get install -y python3-dev build-essential

RUN mkdir -p /usr/src/skin_cancer
WORKDIR /usr/src/skin_cancer

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "skin_cancer.app:app"]
