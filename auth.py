import bcrypt
import secrets
import os
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from models import User
from database import DatabaseService

# Get session timeout from environment
SESSION_TIMEOUT_DAYS = int(os.getenv("SESSION_TIMEOUT_DAYS", 1))

class AuthService:
    """Authentication service for user login and session management"""

    def __init__(self, db_service: Optional[DatabaseService] = None):
        self.db_service = db_service

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except (ValueError, TypeError):
            return False

    def create_session_token(self, user: User, remember_me: bool = False) -> str:
        """Create a new session token for the user"""
        token = secrets.token_urlsafe(32)

        # Set expiration based on remember_me preference
        if remember_me:
            expires_at = datetime.utcnow() + timedelta(days=SESSION_TIMEOUT_DAYS)
        else:
            expires_at = datetime.utcnow() + timedelta(days=1)

        # Store session in database if available, otherwise fall back to in-memory
        if self.db_service:
            self.db_service.create_session(
                user_id=str(user.id),
                token=token,
                expires_at=expires_at,
                remember_me=remember_me
            )
        else:
            # Fallback to in-memory storage for backward compatibility
            if not hasattr(self, '_sessions'):
                self._sessions = {}
            self._sessions[token] = {
                'user_id': str(user.id),
                'username': user.username,
                'display_name': user.display_name,
                'is_admin': user.is_admin,
                'created_at': datetime.utcnow(),
                'expires_at': expires_at,
                'remember_me': remember_me
            }

        return token

    def get_session(self, token: str) -> Optional[Dict]:
        """Get session information by token"""
        if not token:
            return None

        # Try database first if available
        if self.db_service:
            session = self.db_service.get_session_by_token(token)
            if session:
                # Check if session has expired
                if datetime.utcnow() > session['expires_at']:
                    self.db_service.delete_session(token)
                    return None

                # Convert to expected format for compatibility
                return {
                    'user_id': session['user_id'],
                    'username': None,  # Will be populated by verify_session in app.py
                    'display_name': None,  # Will be populated by verify_session in app.py
                    'is_admin': None,  # Will be populated by verify_session in app.py
                    'created_at': session['created_at'],
                    'expires_at': session['expires_at'],
                    'remember_me': session['remember_me']
                }
            return None

        # Fallback to in-memory storage
        if not hasattr(self, '_sessions') or token not in self._sessions:
            return None

        session = self._sessions[token]

        # Check if session has expired
        if datetime.utcnow() > session['expires_at']:
            self.invalidate_session(token)
            return None

        return session

    def invalidate_session(self, token: str) -> bool:
        """Invalidate a session token"""
        # Try database first if available
        if self.db_service:
            return self.db_service.delete_session(token)

        # Fallback to in-memory storage
        if hasattr(self, '_sessions') and token in self._sessions:
            del self._sessions[token]
            return True
        return False

    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        # Try database first if available
        if self.db_service:
            return self.db_service.cleanup_expired_sessions()

        # Fallback to in-memory storage
        if hasattr(self, '_sessions'):
            now = datetime.utcnow()
            expired_tokens = [
                token for token, session in self._sessions.items()
                if now > session['expires_at']
            ]

            for token in expired_tokens:
                del self._sessions[token]
            return len(expired_tokens)
        return 0

    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        # Try database first if available
        if self.db_service:
            # Database cleanup returns count of deleted sessions
            self.cleanup_expired_sessions()
            # Get all active sessions across all users
            from datetime import datetime
            with self.db_service.get_session() as session:
                from models import Session
                count = session.query(Session).filter(
                    Session.expires_at > datetime.utcnow()
                ).count()
                return count

        # Fallback to in-memory storage
        self.cleanup_expired_sessions()
        return len(getattr(self, '_sessions', {}))

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all active sessions for a user"""
        # Try database first if available
        if self.db_service:
            return self.db_service.get_user_sessions(user_id)

        # Fallback to in-memory storage
        self.cleanup_expired_sessions()
        if hasattr(self, '_sessions'):
            return [
                {
                    'token': token,
                    'created_at': session['created_at'],
                    'expires_at': session['expires_at'],
                    'remember_me': session['remember_me']
                }
                for token, session in self._sessions.items()
                if session['user_id'] == user_id
            ]
        return []

    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user"""
        # Try database first if available
        if self.db_service:
            return self.db_service.delete_user_sessions(user_id)

        # Fallback to in-memory storage
        if hasattr(self, '_sessions'):
            tokens_to_remove = [
                token for token, session in self._sessions.items()
                if session['user_id'] == user_id
            ]

            for token in tokens_to_remove:
                del self._sessions[token]

            return len(tokens_to_remove)
        return 0

# Global auth service instance
auth_service = AuthService()