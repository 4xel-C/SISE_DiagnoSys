"""Enforce nom + prenom field being mandatory

Revision ID: f4f97cf5d3cb
Revises: e637b4eaa2de
Create Date: 2026-01-18 17:07:14.155147

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f4f97cf5d3cb"
down_revision: Union[str, Sequence[str], None] = "e637b4eaa2de"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    with op.batch_alter_table("patients") as batch_op:
        batch_op.alter_column(
            "prenom", existing_type=sa.String(length=100), nullable=False
        )
    # ### end Alembic commands ###


def downgrade():
    with op.batch_alter_table("patients") as batch_op:
        batch_op.alter_column(
            "prenom", existing_type=sa.String(length=100), nullable=True
        )

    # ### end Alembic commands ###
