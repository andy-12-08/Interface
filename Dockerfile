FROM python:latest

WORKDIR /app

COPY . .

# Install cmake and build-essential
RUN apt-get update && \
    apt-get install -y cmake \
    build-essential \
    && apt-get clean

# Install pyarrow from pip, which includes pre-built binaries
RUN pip install pyarrow

RUN pip install -r requirements.txt

EXPOSE 8502

ENTRYPOINT [ "streamlit", "run", "--server.port", "8502" ]

CMD ["Home.py"]
