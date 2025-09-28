import os
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from models import User, UserPreference, Base
from auth import AuthService
from database import DatabaseService

class UserService:
    """Service for managing users and their preferences"""

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.auth_service = AuthService()

    def create_user(self, username: str, password: str, display_name: str = None, is_admin: bool = False) -> User:
        """Create a new user"""
        if not username or not password:
            raise ValueError("Username and password are required")

        # Check if username already exists
        if self.get_user_by_username(username):
            raise ValueError(f"Username '{username}' already exists")

        password_hash = self.auth_service.hash_password(password)

        with self.db_service.get_session() as session:
            user = User(
                username=username,
                password_hash=password_hash,
                display_name=display_name or username,
                is_admin=is_admin
            )
            session.add(user)
            session.flush()

            # Create default preferences for the user
            self._create_default_preferences(session, user.id)

            # Detach user from session to avoid session issues
            session.expunge(user)
            return user

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        if not username:
            return None

        with self.db_service.get_session() as session:
            user = session.query(User).filter(
                User.username == username,
                User.is_active == True
            ).first()
            if user:
                session.expunge(user)
            return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        if not user_id:
            return None

        with self.db_service.get_session() as session:
            user = session.query(User).filter(
                User.id == user_id,
                User.is_active == True
            ).first()
            if user:
                session.expunge(user)
            return user

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = self.get_user_by_username(username)
        if not user or not user.password_hash:
            return None

        if self.auth_service.verify_password(password, user.password_hash):
            return user

        return None

    def update_password(self, user_id: str, new_password: str) -> bool:
        """Update user password"""
        if not new_password:
            return False

        password_hash = self.auth_service.hash_password(new_password)

        with self.db_service.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.password_hash = password_hash
                return True
            return False

    def update_user_profile(self, user_id: str, display_name: str = None, avatar_url: str = None, user_context: str = None) -> bool:
        """Update user profile information"""
        with self.db_service.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                if display_name is not None:
                    user.display_name = display_name
                if avatar_url is not None:
                    user.avatar_url = avatar_url
                if user_context is not None:
                    user.user_context = user_context
                return True
            return False

    def update_avatar(self, user_id: str, avatar_url: str) -> bool:
        """Update user avatar"""
        return self.update_user_profile(user_id, avatar_url=avatar_url)

    def update_user_context(self, user_id: str, user_context: str) -> bool:
        """Update user context for LLM"""
        return self.update_user_profile(user_id, user_context=user_context)

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user (admin only)"""
        with self.db_service.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.is_active = False
                return True
            return False

    def activate_user(self, user_id: str) -> bool:
        """Activate a user (admin only)"""
        with self.db_service.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.is_active = True
                return True
            return False

    def delete_user(self, user_id: str) -> bool:
        """Delete a user permanently (admin only)"""
        with self.db_service.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                # Delete user's preferences first
                session.query(UserPreference).filter(UserPreference.user_id == user_id).delete()
                # Delete user's conversations
                from models import Conversation
                session.query(Conversation).filter(Conversation.user_id == user_id).delete()
                # Delete the user
                session.delete(user)
                return True
            return False

    def get_all_users(self) -> List[User]:
        """Get all users (admin only)"""
        with self.db_service.get_session() as session:
            users = session.query(User).order_by(User.created_at.desc()).all()
            # Detach all users from session
            for user in users:
                session.expunge(user)
            return users

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all preferences for a user"""
        with self.db_service.get_session() as session:
            preferences = session.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).all()

            result = {}
            for pref in preferences:
                result[pref.key] = pref.value

            # Ensure default preferences exist
            defaults = self._get_default_preferences()
            for key, default_value in defaults.items():
                if key not in result:
                    result[key] = default_value

            return result

    def set_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Set a user preference"""
        if not key:
            return False

        with self.db_service.get_session() as session:
            # Try to find existing preference
            preference = session.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.key == key
            ).first()

            if preference:
                # Update existing preference
                preference.value = value
            else:
                # Create new preference
                preference = UserPreference(
                    user_id=user_id,
                    key=key,
                    value=value
                )
                session.add(preference)

            return True

    def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a specific user preference"""
        with self.db_service.get_session() as session:
            preference = session.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.key == key
            ).first()

            if preference:
                return preference.value

            # Return default from defaults dict or provided default
            defaults = self._get_default_preferences()
            return defaults.get(key, default)

    def ensure_default_admin(self) -> User:
        """Ensure a default admin user exists"""
        admin_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
        admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")

        # Check if any admin user exists
        with self.db_service.get_session() as session:
            existing_admin = session.query(User).filter(User.is_admin == True).first()
            if existing_admin:
                # Detach from session to avoid session issues
                session.expunge(existing_admin)
                return existing_admin

        # Create default admin user
        try:
            admin_user = self.create_user(
                username=admin_username,
                password=admin_password,
                display_name="Administrator",
                is_admin=True
            )
            return admin_user
        except ValueError:
            # Admin username might exist but not be admin
            with self.db_service.get_session() as session:
                admin_user = session.query(User).filter(User.username == admin_username).first()
                if admin_user:
                    admin_user.is_admin = True
                    session.flush()
                    # Detach from session
                    session.expunge(admin_user)
                    return admin_user
            raise

    def _create_default_preferences(self, session: Session, user_id: str):
        """Create default preferences for a new user"""
        defaults = self._get_default_preferences()

        for key, value in defaults.items():
            preference = UserPreference(
                user_id=user_id,
                key=key,
                value=value
            )
            session.add(preference)

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default preference values"""
        return {
            "last_selected_model": None,
            "sidebar_visible": True,
            "theme": "glass",
            "selected_system_prompt": "General Assistant",
            "custom_system_prompts": {}
        }