FROM ubuntu:20.04

WORKDIR /app

COPY requirements.txt .

ENV DEBIAN_FRONTEND=nonintercative

RUN apt-get update \
    && apt-get -y install python3-pip \
    && apt-get install poppler-utils -y \
    && apt-get install -y ffmpeg libsm6 libxext6 openssl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8052"]