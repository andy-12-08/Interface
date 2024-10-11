FROM python:latest

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8502

ENTRYPOINT [ "streamlit", "run", "--server.port", "8502" ]

CMD ["Home.py"]
