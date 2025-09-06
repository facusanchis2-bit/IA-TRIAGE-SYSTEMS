"""add compound indexes (created_at + predicted_esi, patient_name + created_at)

Revision ID: 0002_add_compound_indexes
Revises: 0001_create_sessions
Create Date: 2025-09-06 00:00:01
"""
from alembic import op

revision = '0002_add_compound_indexes'
down_revision = '0001_create_sessions'
branch_labels = None
depends_on = None

def upgrade():
    # Portables: en SQLite crea índices simples; en Postgres, normales (no concurrently)
    try:
        op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at_esi ON sessions(created_at, predicted_esi)")
    except Exception:
        pass
    try:
        op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_name_created ON sessions(patient_name, created_at)")
    except Exception:
        pass

def downgrade():
    try:
        op.execute("DROP INDEX IF EXISTS idx_sessions_name_created")
    except Exception:
        pass
    try:
        op.execute("DROP INDEX IF EXISTS idx_sessions_created_at_esi")
    except Exception:
        pass
