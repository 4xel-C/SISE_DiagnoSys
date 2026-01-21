"""merge_heads

Revision ID: 8d8067aec651
Revises: b560e8eb7618, f4f97cf5d3cb
Create Date: 2026-01-21 13:11:24.380014

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "8d8067aec651"
down_revision: Union[str, Sequence[str], None] = ("b560e8eb7618", "f4f97cf5d3cb")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
