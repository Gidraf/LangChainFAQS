from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
import uuid
from datetime import datetime
from . import db

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String, unique=True, nullable=False, default=str(uuid.uuid4()))
    message = db.Column(JSONB)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<YourModel id={self.id} session_id={self.session_id} message={self.message} created_at={self.created_at}>"