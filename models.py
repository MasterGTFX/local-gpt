from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")

class Message(Base):
    __tablename__ = 'messages'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'))
    message_data = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Vector(1536), nullable=True)

    conversation = relationship("Conversation", back_populates="messages")

    @property
    def role(self) -> str:
        return self.message_data.get('role', '')

    @property
    def content_text(self) -> str:
        content = self.message_data.get('content', '')
        if isinstance(content, str):
            return content
        return ' '.join(part.get('text', '') for part in content
                       if isinstance(part, dict) and part.get('type') == 'text')

    @property
    def model(self) -> str:
        return self.message_data.get('model', '')

    @property
    def tokens_used(self) -> int:
        return self.message_data.get('usage', {}).get('total_tokens', 0)