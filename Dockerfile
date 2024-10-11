FROM python:3.10

WORKDIR /app

COPY . .

# Install cmake and build-essential
RUN apt-get update && \
    apt-get install -y cmake \
    build-essential \
    && apt-get clean

# Install pyarrow from pip, which includes pre-built binaries
RUN pip install pyarrow

# Install remaining dependencies
RUN pip install -r requirements.txt

EXPOSE 8503

ENTRYPOINT [ "streamlit", "run", "--server.port", "8503" ]
CMD ["Data Source.py"]