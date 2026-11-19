# Slim image for the dashboard (LITE mode: BM25 search, no torch) - fits small hosts like Render free.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURNSIGHT_LITE=1 \
    CHURNSIGHT_PROCESSED_DIR=data/app \
    PORT=8501

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts
COPY docs ./docs
COPY data/app ./data/app
COPY .streamlit ./.streamlit
RUN pip install --no-deps -e .

EXPOSE 8501
CMD ["sh", "-c", "streamlit run app/dashboard.py --server.port ${PORT} --server.address 0.0.0.0"]
