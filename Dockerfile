FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copiamos el proyecto completo (si hay requirements.txt, ahora sí estará disponible)
COPY . .

# Instalación de dependencias:
# - Si existe requirements.txt lo usamos
# - Si no existe, instalamos un set mínimo
RUN python -m pip install -U pip wheel && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir fastapi "uvicorn[standard]" pandas openpyxl sqlalchemy alembic pydantic "psycopg[binary]"; \
    fi

EXPOSE 8000

ENV ADMIN_USER=admin \
    ADMIN_PASS=admin \
    PHI_SALT=cambiame-por-una-sal-larga \
    DB_POOL_SIZE=5 \
    DB_MAX_OVERFLOW=5 \
    DB_POOL_RECYCLE=1800 \
    DB_POOL_TIMEOUT=10

HEALTHCHECK --interval=10s --timeout=5s --retries=10 \
  CMD wget -qO- http://localhost:8000/health >/dev/null 2>&1 || exit 1

CMD ["uvicorn","triage_ai_microservice:app","--host","0.0.0.0","--port","8000"]
