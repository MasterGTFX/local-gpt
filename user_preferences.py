import json
import os
import tempfile
from typing import Dict, Any, Optional
from threading import Lock

class UserPreferences:
    """Manages user preferences stored in a JSON file"""

    def __init__(self, preferences_file: str = "user_preferences.json"):
        self.preferences_file = preferences_file
        self._lock = Lock()
        self._preferences = {}
        self._defaults = {
            "last_selected_model": None,
            "sidebar_visible": True,
            "theme_preference": "light"
        }
        self.load_preferences()

    def get_default_preferences(self) -> Dict[str, Any]:
        """Get default preference values"""
        return self._defaults.copy()

    def load_preferences(self) -> Dict[str, Any]:
        """Load preferences from JSON file with fallback to defaults"""
        with self._lock:
            try:
                if os.path.exists(self.preferences_file):
                    with open(self.preferences_file, 'r', encoding='utf-8') as f:
                        loaded_prefs = json.load(f)

                    # Merge with defaults to ensure all keys exist
                    self._preferences = self._defaults.copy()
                    self._preferences.update(loaded_prefs)
                else:
                    # File doesn't exist, use defaults
                    self._preferences = self._defaults.copy()

            except (json.JSONDecodeError, IOError, OSError) as e:
                print(f"Warning: Could not load preferences from {self.preferences_file}: {e}")
                print("Using default preferences.")
                self._preferences = self._defaults.copy()

        return self._preferences.copy()

    def save_preferences(self) -> bool:
        """Save preferences to JSON file atomically"""
        with self._lock:
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

# Global instance for the application
preferences = UserPreferences()

# Convenience functions for common operations
def get_preference(key: str, default: Any = None) -> Any:
    """Get a preference value"""
    return preferences.get(key, default)

def set_preference(key: str, value: Any) -> bool:
    """Set a preference value"""
    return preferences.set(key, value)

def update_preferences(preferences_dict: Dict[str, Any]) -> bool:
    """Update multiple preferences"""
    return preferences.update(preferences_dict)

def get_all_preferences() -> Dict[str, Any]:
    """Get all preferences"""
    return preferences.get_all()

def load_preferences() -> Dict[str, Any]:
    """Reload preferences from file"""
    return preferences.load_preferences()