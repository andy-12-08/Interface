FROM python:latest

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8502

ENTRYPOINT [ "streamlit", "run", "--server.port", "8502" ]

CMD ["Home.py"]
