from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text, Boolean, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=True)  # nullable for default user
    password_hash = Column(String(255))
    display_name = Column(String(100))
    avatar_url = Column(String(500), nullable=True)  # Path to uploaded avatar image
    user_context = Column(Text, nullable=True)  # User description for LLM context
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user")
    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = 'sessions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    remember_me = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")

class UserPreference(Base):
    __tablename__ = 'user_preferences'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="preferences")

    __table_args__ = (UniqueConstraint('user_id', 'key', name='unique_user_preference'),)

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)  # nullable for backward compatibility
    title = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
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


class FileAttachment(Base):
    __tablename__ = 'file_attachments'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.id'))
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    mime_type = Column(String(100))
    extension = Column(String(10))
    markdown_content = Column(Text)
    content_length = Column(Integer, default=0)
    processing_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

    message = relationship("Message", backref="file_attachments")