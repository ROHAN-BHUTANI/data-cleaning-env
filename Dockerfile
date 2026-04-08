FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pandas pydantic openai

COPY server/ /app/
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
