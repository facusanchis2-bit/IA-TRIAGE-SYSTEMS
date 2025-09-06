from __future__ import annotations
import os
from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context

# Config de Alembic
config = context.config

# Tomamos DATABASE_URL del entorno (prioritario) o del alembic.ini (DB_URL)
db_url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url").replace("%(DB_URL)s", "")
if not db_url:
    # Fallback a SQLite local
    from pathlib import Path
    db_url = f"sqlite:///{Path(__file__).resolve().parents[1] / 'triage_sessions.db'}"

# logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None  # usamos migraciones **imperativas** (op.execute), no autogenerate

def run_migrations_offline():
    context.configure(
        url=db_url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = create_engine(db_url, poolclass=pool.NullPool, future=True)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
