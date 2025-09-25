from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from models import Base, Conversation, Message, FileAttachment
from typing import List, Tuple, Optional, Dict
import os

class DatabaseService:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def init_db(self):
        """Initialize database and create tables"""
        try:
            # Enable pgvector extension if available
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
        except Exception:
            # Skip if pgvector is not available
            pass

        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_conversation(self, title: str = None) -> str:
        """Create a new conversation and return its ID"""
        with self.get_session() as session:
            conversation = Conversation(title=title)
            session.add(conversation)
            session.flush()
            return str(conversation.id)

    def save_message(self, conversation_id: str, role: str, content, model: str = None,
                    usage: dict = None, **metadata) -> str:
        """Save a message to the conversation"""
        message_data = {
            'role': role,
            'content': content,
            'model': model,
            'usage': usage or {},
            'metadata': metadata
        }

        with self.get_session() as session:
            message = Message(conversation_id=conversation_id, message_data=message_data)
            session.add(message)
            session.flush()

            # Update conversation timestamp
            conversation = session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                conversation.updated_at = message.created_at

            return str(message.id)

    def get_conversation_messages(self, conversation_id: str) -> List[Tuple[str, str]]:
        """Get all messages for a conversation in Gradio format"""
        with self.get_session() as session:
            messages = session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).all()

            # Convert to Gradio format: [(user_msg, assistant_msg), ...]
            gradio_history = []
            user_msg = None

            for message in messages:
                if message.role == 'user':
                    user_msg = message.content_text
                elif message.role == 'assistant' and user_msg is not None:
                    gradio_history.append((user_msg, message.content_text))
                    user_msg = None

            return gradio_history

    def get_recent_conversations(self, limit: int = 10) -> List[dict]:
        """Get recent conversations with basic info"""
        with self.get_session() as session:
            conversations = session.query(Conversation).order_by(
                Conversation.updated_at.desc()
            ).limit(limit).all()

            result = []
            for conv in conversations:
                # Get first message for title if no title set
                title = conv.title
                if not title and conv.messages:
                    first_msg = conv.messages[0]
                    if first_msg.role == 'user':
                        title = first_msg.content_text[:50] + "..." if len(first_msg.content_text) > 50 else first_msg.content_text

                result.append({
                    'id': str(conv.id),
                    'title': title or 'New Conversation',
                    'updated_at': conv.updated_at
                })

            return result

    def search_conversations(self, query: str, limit: int = 10) -> List[dict]:
        """Search conversations by title and message content"""
        if not query or not query.strip():
            return self.get_recent_conversations(limit)

        query = query.strip().lower()

        with self.get_session() as session:
            # Search in conversation titles
            title_matches = session.query(Conversation).filter(
                Conversation.title.ilike(f'%{query}%')
            ).order_by(Conversation.updated_at.desc()).limit(limit).all()

            # Search in message content using JSONB operators
            content_matches = session.query(Conversation).join(Message).filter(
                Message.message_data['content'].astext.ilike(f'%{query}%')
            ).distinct().order_by(Conversation.updated_at.desc()).limit(limit).all()

            # Combine results and remove duplicates
            all_conversations = {}
            for conv in title_matches + content_matches:
                all_conversations[conv.id] = conv

            # Sort by updated_at and limit results
            sorted_conversations = sorted(
                all_conversations.values(),
                key=lambda x: x.updated_at,
                reverse=True
            )[:limit]

            result = []
            for conv in sorted_conversations:
                # Get first message for title if no title set
                title = conv.title
                if not title and conv.messages:
                    first_msg = conv.messages[0]
                    if first_msg.role == 'user':
                        title = first_msg.content_text[:50] + "..." if len(first_msg.content_text) > 50 else first_msg.content_text

                result.append({
                    'id': str(conv.id),
                    'title': title or 'New Conversation',
                    'updated_at': conv.updated_at
                })

            return result

    def set_conversation_title(self, conversation_id: str, title: str):
        """Set the title for a conversation"""
        with self.get_session() as session:
            conversation = session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                conversation.title = title

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages"""
        with self.get_session() as session:
            # Delete messages first (due to foreign key constraint)
            session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).delete()

            # Delete conversation
            session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).delete()

    def save_file_attachment(self, message_id: str, file_data: Dict) -> str:
        """Save a file attachment to the database"""
        with self.get_session() as session:
            attachment = FileAttachment(
                message_id=message_id,
                filename=file_data['file_info']['filename'],
                original_filename=file_data['file_info']['filename'],
                file_size=file_data['file_info']['size'],
                file_hash=file_data['file_info']['hash'],
                mime_type=file_data['file_info']['mime_type'],
                extension=file_data['file_info']['extension'],
                markdown_content=file_data['markdown_content'],
                content_length=file_data.get('content_length', 0),
                processing_metadata={'success': file_data['success'], 'error': file_data.get('error')}
            )
            session.add(attachment)
            session.flush()
            return str(attachment.id)

    def get_message_attachments(self, message_id: str) -> List[Dict]:
        """Get all file attachments for a message"""
        with self.get_session() as session:
            attachments = session.query(FileAttachment).filter(
                FileAttachment.message_id == message_id
            ).all()

            return [{
                'id': str(att.id),
                'filename': att.filename,
                'original_filename': att.original_filename,
                'file_size': att.file_size,
                'file_hash': att.file_hash,
                'mime_type': att.mime_type,
                'extension': att.extension,
                'markdown_content': att.markdown_content,
                'content_length': att.content_length,
                'processing_metadata': att.processing_metadata,
                'created_at': att.created_at
            } for att in attachments]

    def check_file_exists(self, file_hash: str) -> Optional[Dict]:
        """Check if a file with the given hash already exists"""
        with self.get_session() as session:
            attachment = session.query(FileAttachment).filter(
                FileAttachment.file_hash == file_hash
            ).first()

            if attachment:
                return {
                    'id': str(attachment.id),
                    'filename': attachment.filename,
                    'file_size': attachment.file_size,
                    'markdown_content': attachment.markdown_content,
                    'created_at': attachment.created_at
                }
            return None

    def get_conversation_attachments(self, conversation_id: str) -> List[Dict]:
        """Get all file attachments for a conversation"""
        with self.get_session() as session:
            attachments = session.query(FileAttachment).join(Message).filter(
                Message.conversation_id == conversation_id
            ).all()

            return [{
                'id': str(att.id),
                'filename': att.filename,
                'file_size': att.file_size,
                'mime_type': att.mime_type,
                'extension': att.extension,
                'created_at': att.created_at,
                'message_id': str(att.message_id)
            } for att in attachments]