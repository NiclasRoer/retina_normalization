FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY pyproject.toml README.md ./
COPY packages ./packages
COPY apps ./apps

RUN pip install --no-cache-dir .

CMD ["python", "apps/main.py"]
