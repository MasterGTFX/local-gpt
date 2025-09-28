import json
import os
import tempfile
from typing import Dict, Any, Optional
from threading import Lock

class UserPreferences:
    """Manages user preferences - supports both file-based and database-backed storage"""

    def __init__(self, preferences_file: str = "user_preferences.json", user_service=None, user_id: str = None):
        self.preferences_file = preferences_file
        self.user_service = user_service
        self.user_id = user_id
        self._lock = Lock()
        self._preferences = {}
        self._defaults = {
            "last_selected_model": None,
            "sidebar_visible": True,
            "theme": "glass",
            "selected_system_prompt": "General Assistant",
            "custom_system_prompts": {}
        }
        self.load_preferences()

    def get_default_preferences(self) -> Dict[str, Any]:
        """Get default preference values"""
        return self._defaults.copy()

    def load_preferences(self) -> Dict[str, Any]:
        """Load preferences from database or JSON file with fallback to defaults"""
        with self._lock:
            try:
                # If user service and user ID are available, use database
                if self.user_service and self.user_id:
                    self._preferences = self.user_service.get_user_preferences(self.user_id)
                else:
                    # Fallback to file-based preferences
                    self._load_from_file()

            except Exception as e:
                print(f"Warning: Could not load preferences: {e}")
                print("Using default preferences.")
                self._preferences = self._defaults.copy()

        return self._preferences.copy()

    def _load_from_file(self):
        """Load preferences from JSON file"""
        if os.path.exists(self.preferences_file):
            with open(self.preferences_file, 'r', encoding='utf-8') as f:
                loaded_prefs = json.load(f)

            # Merge with defaults to ensure all keys exist
            self._preferences = self._defaults.copy()
            self._preferences.update(loaded_prefs)
        else:
            # File doesn't exist, use defaults
            self._preferences = self._defaults.copy()

    def save_preferences(self) -> bool:
        """Save preferences to database or JSON file"""
        with self._lock:
            try:
                # If user service and user ID are available, save to database
                if self.user_service and self.user_id:
                    return self._save_to_database()
                else:
                    # Fallback to file-based preferences
                    return self._save_to_file()

            except Exception as e:
                print(f"Warning: Could not save preferences: {e}")
                return False

    def _save_to_database(self) -> bool:
        """Save preferences to database"""
        try:
            for key, value in self._preferences.items():
                self.user_service.set_user_preference(self.user_id, key, value)
            return True
        except Exception as e:
            print(f"Warning: Could not save preferences to database: {e}")
            return False

    def _save_to_file(self) -> bool:
        """Save preferences to JSON file atomically"""
        try:
            # Use atomic write to prevent corruption
            temp_dir = os.path.dirname(os.path.abspath(self.preferences_file))
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=temp_dir,
                delete=False,
                suffix='.tmp'
            ) as temp_file:
                json.dump(self._preferences, temp_file, indent=2)
                temp_filename = temp_file.name

            # Atomic move on Unix systems
            os.replace(temp_filename, self.preferences_file)
            return True

        except (IOError, OSError, json.JSONEncodeError) as e:
            print(f"Warning: Could not save preferences to {self.preferences_file}: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_filename' in locals():
                    os.unlink(temp_filename)
            except:
                pass
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference value"""
        with self._lock:
            return self._preferences.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set a preference value and save to file"""
        with self._lock:
            self._preferences[key] = value
        return self.save_preferences()

    def update(self, preferences_dict: Dict[str, Any]) -> bool:
        """Update multiple preferences at once"""
        with self._lock:
            self._preferences.update(preferences_dict)
        return self.save_preferences()

    def get_all(self) -> Dict[str, Any]:
        """Get all preferences"""
        with self._lock:
            return self._preferences.copy()

    def reset_to_defaults(self) -> bool:
        """Reset all preferences to default values"""
        with self._lock:
            self._preferences = self._defaults.copy()
        return self.save_preferences()

# Global instance for the application (fallback mode)
preferences = UserPreferences()

# Active preferences instance (can be overridden for multi-user)
_active_preferences = preferences

def set_active_preferences(prefs: UserPreferences):
    """Set the active preferences instance for multi-user support"""
    global _active_preferences
    _active_preferences = prefs

def get_active_preferences() -> UserPreferences:
    """Get the current active preferences instance"""
    return _active_preferences

# Convenience functions for common operations
def get_preference(key: str, default: Any = None) -> Any:
    """Get a preference value"""
    return _active_preferences.get(key, default)

def set_preference(key: str, value: Any) -> bool:
    """Set a preference value"""
    return _active_preferences.set(key, value)

def update_preferences(preferences_dict: Dict[str, Any]) -> bool:
    """Update multiple preferences"""
    return _active_preferences.update(preferences_dict)

def get_all_preferences() -> Dict[str, Any]:
    """Get all preferences"""
    return _active_preferences.get_all()

def load_preferences() -> Dict[str, Any]:
    """Reload preferences from file or database"""
    return _active_preferences.load_preferences()

def create_user_preferences(user_service, user_id: str) -> UserPreferences:
    """Create a new UserPreferences instance for a specific user"""
    return UserPreferences(user_service=user_service, user_id=user_id)