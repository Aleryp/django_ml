FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

RUN chmod 777 ./entrypoint.sh

CMD ["./entrypoint.sh"]