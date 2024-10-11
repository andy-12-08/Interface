FROM python:latest

WORKDIR /app

COPY . .

# Install dependencies for building pyarrow
RUN apt-get update && \
    apt-get install -y cmake \
    libarrow-dev \
    libparquet-dev \
    build-essential \
    && apt-get clean


RUN pip install -r requirements.txt

EXPOSE 8502

ENTRYPOINT [ "streamlit", "run", "--server.port", "8502" ]

CMD ["Home.py"]
