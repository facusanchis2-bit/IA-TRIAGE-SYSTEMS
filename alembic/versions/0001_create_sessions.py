"""create sessions table + indexes

Revision ID: 0001_create_sessions
Revises: 
Create Date: 2025-09-06 00:00:00
"""
from alembic import op
import sqlalchemy as sa

# Revisiones
revision = '0001_create_sessions'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name  # 'postgresql' o 'sqlite'

    # Crear tabla si no existe (portable)
    op.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TEXT,
        patient_json TEXT,
        patient_name TEXT,
        patient_hash TEXT,
        chief_complaint TEXT,
        extracted_entities_json TEXT,
        derived_features_json TEXT,
        predicted_esi INTEGER,
        mapped_manchester TEXT,
        recommendation TEXT,
        override_applied INTEGER,
        override_reason TEXT,
        differentials_json TEXT,
        shap_top_json TEXT,
        llm_summary_json TEXT,
        disclaimer_ack INTEGER,
        state_json TEXT
    )
    """)

    # Índices (try/except para idempotencia)
    # created_at para ordenar/filtrar por fecha
    try:
        op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)")
    except Exception:
        pass

    # patient_name para filtros por nombre
    try:
        op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_patient_name ON sessions(patient_name)")
    except Exception:
        pass

    # predicted_esi para filtros por ESI
    try:
        op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_predicted_esi ON sessions(predicted_esi)")
    except Exception:
        pass

def downgrade():
    # Bajada conservadora: solo dropear índices (no borramos la tabla en producción)
    try:
        op.execute("DROP INDEX IF EXISTS idx_sessions_predicted_esi")
    except Exception:
        pass
    try:
        op.execute("DROP INDEX IF EXISTS idx_sessions_patient_name")
    except Exception:
        pass
    try:
        op.execute("DROP INDEX IF EXISTS idx_sessions_created_at")
    except Exception:
        pass
    # Si quisieras dropear la tabla (no recomendado):
    # op.execute("DROP TABLE IF EXISTS sessions")
