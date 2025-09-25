"""
Configuration for Local GPT Chat
Customize your deployment by editing the values below.
"""

import gradio as gr
from user_preferences import get_preference


# =============================================================================
# APP IDENTITY & BRANDING
# =============================================================================

APP_TITLE = "Local GPT Chat"
APP_EMOJI = "🤖"
APP_DESCRIPTION = "Private AI conversations on your Raspberry Pi"

# Customize the header text
def get_header_html() -> str:
    return f"<h1 style='text-align: center;'>{APP_EMOJI} {APP_TITLE}</h1>"


# =============================================================================
# THEME CONFIGURATION
# =============================================================================

DEFAULT_THEME = "Glass"

def get_theme() -> gr.Theme:
    """Get Gradio theme from user preference"""
    theme_name = get_preference("theme", DEFAULT_THEME)
    try:
        theme_class = getattr(gr.themes, theme_name)
        return theme_class()
    except AttributeError:
        return getattr(gr.themes, DEFAULT_THEME)()


# =============================================================================
# CHAT INTERFACE SETTINGS
# =============================================================================

CHAT_CONFIG = {
    "height": 600,
    "show_copy_button": True,
    "enable_editing": True,        # Allow users to edit messages
    "enable_retry": True,          # Allow regenerating responses
    "enable_undo": True,           # Allow undoing messages
    "show_label": False,
    "container": False,
}

def get_chat_config() -> dict:
    """Get chatbot configuration"""
    return {
        "type": "messages",
        "value": [],
        "height": CHAT_CONFIG["height"],
        "show_label": CHAT_CONFIG["show_label"],
        "container": CHAT_CONFIG["container"],
        "editable": "all" if CHAT_CONFIG["enable_editing"] else False,
        "show_copy_button": CHAT_CONFIG["show_copy_button"]
    }


# =============================================================================
# SIDEBAR & NAVIGATION
# =============================================================================

SIDEBAR_CONFIG = {
    "min_width": 250,
    "default_visible": True,
    "enable_search": True,
    "max_conversations": 20,
    "conversation_title_max_length": 25,
}


# =============================================================================
# MODEL SELECTION
# =============================================================================

MODEL_CONFIG = {
    "show_pricing": True,           # Show pricing info in model dropdown
    "show_model_info": True,        # Show expandable model information
    "cache_models": True,           # Cache model list for performance
    "sort_by_date": True,           # Sort models by creation date
}


# =============================================================================
# FEATURES & TOGGLES
# =============================================================================

FEATURES = {
    # Core features
    "conversation_persistence": True,    # Save/load conversations (requires DB)
    "conversation_search": True,         # Search through conversations
    "model_switching": True,             # Allow changing models mid-conversation

    # UI features
    "dark_mode_toggle": False,           # Show theme toggle button (disabled - use config)
    "sidebar_toggle": True,              # Allow hiding/showing sidebar
    "clear_chat": True,                  # Show clear conversation button
    "new_chat": True,                    # Show new chat button

    # Advanced features
    "message_editing": True,             # Edit messages after sending
    "conversation_branching": True,      # Retry/regenerate responses
    "export_conversations": False,       # Export chat history (not implemented)
}


# =============================================================================
# PERFORMANCE & LIMITS
# =============================================================================

LIMITS = {
    "max_message_length": 10000,        # Max characters per message
    "max_conversation_length": 100,     # Max messages per conversation
    "api_timeout": 30,                   # API request timeout (seconds)
    "model_cache_ttl": 3600,            # Model list cache time (seconds)
}


# =============================================================================
# DEFAULT USER PREFERENCES
# =============================================================================

DEFAULT_PREFERENCES = {
    "last_selected_model": None,
    "sidebar_visible": SIDEBAR_CONFIG["default_visible"],
    "theme": DEFAULT_THEME,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    return FEATURES.get(feature_name, False)

def get_limit(limit_name: str) -> int:
    """Get a limit value"""
    return LIMITS.get(limit_name, 0)

def get_sidebar_config(key: str = None):
    """Get sidebar configuration"""
    if key:
        return SIDEBAR_CONFIG.get(key)
    return SIDEBAR_CONFIG

def get_model_config(key: str = None):
    """Get model configuration"""
    if key:
        return MODEL_CONFIG.get(key)
    return MODEL_CONFIG