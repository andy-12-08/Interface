FROM python:latest

WORKDIR /app

COPY . .

# Add this to install cmake
RUN apt-get update && apt-get install -y cmake

RUN pip install -r requirements.txt

EXPOSE 8502

ENTRYPOINT [ "streamlit", "run", "--server.port", "8502" ]

CMD ["Home.py"]
