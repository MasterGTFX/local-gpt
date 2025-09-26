import bcrypt
import secrets
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from models import User

class AuthService:
    """Authentication service for user login and session management"""

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}

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
            expires_at = datetime.utcnow() + timedelta(days=30)
        else:
            expires_at = datetime.utcnow() + timedelta(hours=24)

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
        if not token or token not in self._sessions:
            return None

        session = self._sessions[token]

        # Check if session has expired
        if datetime.utcnow() > session['expires_at']:
            self.invalidate_session(token)
            return None

        return session

    def invalidate_session(self, token: str) -> bool:
        """Invalidate a session token"""
        if token in self._sessions:
            del self._sessions[token]
            return True
        return False

    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory"""
        now = datetime.utcnow()
        expired_tokens = [
            token for token, session in self._sessions.items()
            if now > session['expires_at']
        ]

        for token in expired_tokens:
            del self._sessions[token]

    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        self.cleanup_expired_sessions()
        return len(self._sessions)

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all active sessions for a user"""
        self.cleanup_expired_sessions()
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

    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user"""
        tokens_to_remove = [
            token for token, session in self._sessions.items()
            if session['user_id'] == user_id
        ]

        for token in tokens_to_remove:
            del self._sessions[token]

        return len(tokens_to_remove)

# Global auth service instance
auth_service = AuthService()