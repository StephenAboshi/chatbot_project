"""Update metrics and scenarios

Revision ID: 9475e27cbc14
Revises: 
Create Date: 2024-08-26 21:09:09.976925

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9475e27cbc14'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('username', sa.VARCHAR(length=50), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username')
    )
    # ### end Alembic commands ###
