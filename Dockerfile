# ---- 1. build the React control tower ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python API serving the built SPA ----
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONWARNINGS=ignore
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt
COPY backend/ backend/
COPY ml/ ml/
COPY models/ models/
COPY experiments/ experiments/
COPY datasets/synthetic/ datasets/synthetic/
COPY datasets/experiment/ datasets/experiment/
COPY --from=frontend /fe/dist frontend/dist
# Build the demo database at image build time (SQLite unless DATABASE_URL is set at runtime)
RUN python -m backend.database_setup --drop && python -m backend.data_import
EXPOSE 10000
CMD ["sh", "-c", "if [ -n \"$DATABASE_URL\" ]; then python -m backend.data_import; fi; exec gunicorn -w 1 --threads 4 --timeout 300 -b 0.0.0.0:${PORT:-10000} backend.run:app"]
