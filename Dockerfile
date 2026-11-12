FROM python:3.13-slim

# libgomp is needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=5000
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",5000)}/health')"
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT}"]
