import os
import requests
import gradio as gr
import threading
import time
import base64
import mimetypes
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from database import DatabaseService
from user_preferences import get_preference, set_preference, load_preferences, create_user_preferences, set_active_preferences
from file_processor import FileProcessor
from user_service import UserService
from auth import auth_service, AuthService
from system_prompts import get_predefined_prompts, get_prompt_by_display_name, get_display_names

# Load environment variables
load_dotenv()

# Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
PORT = int(os.getenv("PORT", 7860))
GRADIO_THEME = os.getenv("GRADIO_THEME", "glass")

# Authentication configuration
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
SESSION_TIMEOUT_DAYS = int(os.getenv("SESSION_TIMEOUT_DAYS", 1))

# Global variables
available_models = []
selected_model = None
db_service: Optional[DatabaseService] = None
user_service: Optional[UserService] = None
current_conversation_id: Optional[str] = None
current_model_choice = ""
user_preferences = {}
file_processor: Optional[FileProcessor] = None
current_conversation_tokens = 0
current_user = None
last_conversations_user_id = None  # Track the user for whom conversations were last loaded

def fetch_available_models() -> List[Dict]:
    """Fetch available models from LLM API"""
    try:
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.get(f"{LLM_BASE_URL}/models", headers=headers)
        response.raise_for_status()

        models_data = response.json()
        models = models_data.get("data", [])

        # Sort models by creation date (newest first)
        models.sort(key=lambda x: x.get("created", 0), reverse=True)

        return models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def get_model_choices(filter_free: bool = False) -> List[str]:
    """Get list of model names for dropdown"""
    global available_models
    if not available_models:
        available_models = fetch_available_models()

    if not available_models:
        return ["No models available"]

    # Create user-friendly model names with pricing info
    choices = []
    for model in available_models:
        name = model.get("name", "Unknown")
        model_id = model.get("id", "")
        pricing = model.get("pricing", {})

        # Check if this is a free model (contains "free" in name)
        is_free_model = "free" in name.lower()

        # Skip if we're filtering for free models and this isn't free
        if filter_free and not is_free_model:
            continue

        # Get pricing info per 1M tokens
        prompt_price = float(pricing.get("prompt", "0")) * 1_000_000
        completion_price = float(pricing.get("completion", "0")) * 1_000_000
        image_price = float(pricing.get("image", "0"))

        # Build pricing string
        price_parts = []
        if prompt_price > 0:
            price_parts.append(f"${prompt_price:.2f}")
        if completion_price > 0 and completion_price != prompt_price:
            price_parts.append(f"${completion_price:.2f}")
        if image_price > 0:
            price_parts.append(f"${image_price:.3f} (img)")

        if price_parts:
            price_str = "/".join(price_parts)
            choice = f"{name} - {price_str}"
        else:
            choice = name

        choices.append(choice)

    return choices if choices else ["No models available"]

def extract_model_id(model_choice: str) -> str:
    """Extract model ID from dropdown choice"""
    # Extract model name from the first line (before newline or dash)
    model_name = model_choice.split("\n")[0] if "\n" in model_choice else model_choice.split(" - ")[0]

    for model in available_models:
        if model.get("name", "") == model_name:
            return model.get("id", "")

    # Fallback: return the model name
    return model_name

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def create_image_message_content(text: str, image_path: str) -> List[Dict]:
    """Create message content with both text and image for OpenAI API"""
    content = []

    if text and text.strip():
        content.append({
            "type": "text",
            "text": text
        })

    if image_path:
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = "image/jpeg"  # Default fallback

        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if base64_image:
            data_url = f"data:{mime_type};base64,{base64_image}"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })

    return content

def find_model_choice_by_id(model_id: str, model_choices: List[str]) -> Optional[str]:
    """Find the model choice string that corresponds to a given model ID"""
    if not model_id or not model_choices:
        return None

    for choice in model_choices:
        if extract_model_id(choice) == model_id:
            return choice

    return None

def messages_to_openai_format(messages: List[Dict]) -> List[Dict]:
    """Convert Gradio messages format to OpenAI API format"""
    formatted_messages = []
    for msg in messages:
        if msg.get("content"):
            # Handle both string content and structured content (for images)
            formatted_message = {"role": msg["role"], "content": msg["content"]}
            formatted_messages.append(formatted_message)
    return formatted_messages

def send_message_to_llm(messages: List[Dict[str, str]], model_id: str, selected_system_prompt: str = None) -> Tuple[str, Optional[Dict]]:
    """Send message to LLM API and get response with token usage"""
    try:
        # Build system message combining system prompt and user context
        enhanced_messages = messages.copy()
        system_parts = []

        # Add selected system prompt
        if selected_system_prompt:
            # Get the actual prompt text
            if selected_system_prompt.startswith("Custom: "):
                # Custom prompt
                custom_name = selected_system_prompt[8:]  # Remove "Custom: " prefix
                custom_prompts = get_preference("custom_system_prompts", {})
                prompt_text = custom_prompts.get(custom_name, "")
            else:
                # Predefined prompt
                prompt_text = get_prompt_by_display_name(selected_system_prompt)

            if prompt_text:
                system_parts.append(prompt_text)

        # Add user context if available
        if current_user and hasattr(current_user, 'user_context') and current_user.user_context:
            system_parts.append(f"User context: {current_user.user_context}")

        # Create system message if we have any system content
        if system_parts:
            system_message = {
                "role": "system",
                "content": "\n\n".join(system_parts)
            }
            enhanced_messages = [system_message] + enhanced_messages

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Local GPT Chat"
        }

        data = {
            "model": model_id,
            "messages": enhanced_messages
        }

        response = requests.post(f"{LLM_BASE_URL}/chat/completions",
                               headers=headers, json=data)
        response.raise_for_status()

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage")
            return content, usage
        else:
            return "Error: No response from model", None

    except Exception as e:
        return f"Error: {str(e)}", None

def ensure_conversation() -> str:
    """Ensure we have a current conversation, create one if needed"""
    global current_conversation_id, db_service, current_user

    if not current_conversation_id and db_service:
        user_id = str(current_user.id) if current_user else None
        current_conversation_id = db_service.create_conversation(user_id=user_id)

    return current_conversation_id

def generate_conversation_title(content: str) -> str:
    """Generate conversation title from first 3 words of user message"""
    if not content or not content.strip():
        return "New Conversation"

    words = content.strip().split()[:3]
    return " ".join(words) if words else "New Conversation"

def load_conversations_list(search_query: str = "") -> List[Tuple[str, str]]:
    """Load conversations for Radio component"""
    global db_service, current_user, last_conversations_user_id

    current_user_id = str(current_user.id) if current_user else None
    print(f"[CONV] Loading conversations list, current_user: {current_user.username if current_user else 'None'}")
    print(f"[CONV] Current user_id: {current_user_id}, last loaded for: {last_conversations_user_id}")

    # Update the last user for whom conversations were loaded
    last_conversations_user_id = current_user_id

    if not db_service:
        print(f"[CONV] No database service available")
        return []

    try:
        user_id = current_user_id
        print(f"[CONV] Using user_id: {user_id}")

        if search_query and search_query.strip():
            print(f"[CONV] Searching conversations with query: '{search_query}'")
            conversations = db_service.search_conversations(search_query, limit=2000, user_id=user_id)
        else:
            print(f"[CONV] Loading recent conversations")
            conversations = db_service.get_recent_conversations(limit=2000, user_id=user_id)

        if not conversations:
            print(f"[CONV] No conversations found")
            return []

        print(f"[CONV] Processing {len(conversations)} conversations for display")

        choices = []
        for conv in conversations:
            title = conv.get('title', 'New Conversation')
            conv_id = conv.get('id', '')
            updated_at = conv.get('updated_at')

            # Format timestamp
            time_str = ""
            if updated_at:
                try:
                    from datetime import datetime
                    if isinstance(updated_at, str):
                        dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    else:
                        dt = updated_at
                    time_str = dt.strftime("%m/%d %H:%M")
                except:
                    time_str = ""

            # Truncate long titles to prevent overflow
            max_title_length = 25
            if len(title) > max_title_length:
                title = title[:max_title_length] + "..."

            # Create compact display name with timestamp
            display_name = f"{title}\n{time_str}" if time_str else title
            choices.append((display_name, conv_id))

        print(f"[CONV] Returning {len(choices)} conversation choices")
        return choices

    except Exception as e:
        print(f"[CONV] Error loading conversations: {e}")
        return []

def load_conversation_messages(conversation_id: str) -> List[Dict[str, str]]:
    """Load conversation messages in Gradio format"""
    global db_service, current_user

    print(f"[CONV] Loading messages for conversation {conversation_id}, current_user: {current_user.username if current_user else 'None'}")

    if not db_service or not conversation_id:
        print(f"[CONV] Missing database service or conversation_id")
        return []

    try:
        # Get user ID for security verification
        user_id = str(current_user.id) if current_user else None
        print(f"[CONV] Loading messages with user_id: {user_id}")

        # Get messages using the existing database method with user verification
        messages = db_service.get_conversation_messages(conversation_id, user_id=user_id)

        if not messages:
            print(f"[CONV] No messages found for conversation {conversation_id}")
            return []

        print(f"[CONV] Converting {len(messages)} message pairs to Gradio format")

        # Convert from old format [(user_msg, assistant_msg)] to new messages format
        gradio_messages = []
        for user_msg, assistant_msg in messages:
            gradio_messages.append({"role": "user", "content": user_msg})
            gradio_messages.append({"role": "assistant", "content": assistant_msg})

        print(f"[CONV] Returning {len(gradio_messages)} messages")
        return gradio_messages

    except Exception as e:
        print(f"[CONV] Error loading conversation messages: {e}")
        return []

def toggle_sidebar_visibility(visible: bool):
    """Toggle sidebar visibility and save preference"""
    new_visible = not visible
    set_preference("sidebar_visible", new_visible)
    return new_visible, new_visible

def start_new_conversation() -> List[Dict[str, str]]:
    """Start a new conversation and return empty chat history"""
    global current_conversation_id, current_conversation_tokens
    current_conversation_id = None  # Will be created when first message is sent
    current_conversation_tokens = 0
    return []

def refresh_conversations_for_user() -> List[Tuple[str, str]]:
    """Refresh conversation list for the current user - used after authentication"""
    print(f"[CONV] Refreshing conversations for current user: {current_user.username if current_user else 'None'}")
    return load_conversations_list("")

def update_model_selection_for_user() -> str:
    """Update model selection based on user's last used model from recent conversations"""
    global current_user, db_service

    if not (AUTH_ENABLED and current_user and db_service):
        return None

    try:
        # Get the last used model ID from user's conversation history
        last_used_model_id = db_service.get_user_last_used_model(str(current_user.id))

        if last_used_model_id:
            # Find the corresponding model choice string
            model_choices = get_model_choices()
            model_choice = find_model_choice_by_id(last_used_model_id, model_choices)

            if model_choice:
                print(f"[MODEL] Found last used model for {current_user.username}: {last_used_model_id}")
                return model_choice
            else:
                print(f"[MODEL] Last used model {last_used_model_id} not found in current choices for {current_user.username}")
        else:
            print(f"[MODEL] No last used model found for {current_user.username}")

    except Exception as e:
        print(f"[MODEL] Error getting last used model for {current_user.username}: {e}")

    return None

def update_system_prompt_selection_for_user() -> str:
    """Update system prompt selection based on user's last used system prompt from recent conversations"""
    global current_user, db_service

    if not (AUTH_ENABLED and current_user and db_service):
        return None

    try:
        # Get the last used system prompt name from user's conversation history
        last_used_system_prompt = db_service.get_user_last_used_system_prompt_name(str(current_user.id))

        if last_used_system_prompt:
            print(f"[SYSTEM_PROMPT] Found last used system prompt for {current_user.username}: {last_used_system_prompt}")
            return last_used_system_prompt
        else:
            print(f"[SYSTEM_PROMPT] No last used system prompt found for {current_user.username}, using default")

    except Exception as e:
        print(f"[SYSTEM_PROMPT] Error getting last used system prompt for {current_user.username}: {e}")

    return "General Assistant"  # Default fallback

def chat_function(message: str, history: List[Dict[str, str]], model_choice: str, system_prompt_choice: str, uploaded_files: List[Dict] = None, image_file = None) -> Tuple[str, List[Dict[str, str]], List[Dict], gr.update, None]:
    """Handle chat messages with optional file attachments"""
    global db_service, current_conversation_id, file_processor

    if not LLM_API_KEY:
        error_msg = {"role": "assistant", "content": "Error: Please set your LLM_API_KEY in the .env file"}
        return "", history + [error_msg], [], gr.update(visible=False, value=[]), None

    if not message.strip() and not image_file:
        return "", history, uploaded_files or [], gr.update(), image_file

    if model_choice == "No models available":
        error_msg = {"role": "assistant", "content": "Error: No models available. Please check your API key."}
        return "", history + [error_msg], [], gr.update(visible=False, value=[]), None

    model_id = extract_model_id(model_choice)

    # Process file attachments if any
    file_context = ""
    processed_files = uploaded_files or []
    if processed_files and file_processor:
        # Format files for LLM context
        file_context = file_processor.format_files_for_llm_context(processed_files)

    # Handle image input if provided
    image_path = None
    if image_file and hasattr(image_file, 'name'):
        image_path = image_file.name

    # Prepare the actual message for the LLM
    if image_path:
        # Create structured content for image + text
        llm_message_content = create_image_message_content(
            f"{file_context}\n\n[USER MESSAGE]\n{message}" if file_context else message,
            image_path
        )
        llm_message = llm_message_content
    elif file_context:
        llm_message = f"{file_context}\n\n[USER MESSAGE]\n{message}"
    else:
        llm_message = message

    # Ensure we have a conversation to save to
    conversation_id = ensure_conversation()

    # If this is the first message in a new conversation, set the title
    if db_service and conversation_id and len(history) == 0:
        title = generate_conversation_title(message)
        try:
            db_service.set_conversation_title(conversation_id, title)
        except Exception as e:
            print(f"Error setting conversation title: {e}")

    # Add user message to history (display the original message, not the one with file context)
    # For display purposes, keep it simple - just the text message
    user_message = {"role": "user", "content": message + (" 📷" if image_path else "")}
    updated_history = history + [user_message]

    # Save user message to database and get message ID for file attachments
    user_message_id = None
    if db_service and conversation_id:
        try:
            user_message_id = db_service.save_message(conversation_id, "user", message, model_id)
        except Exception as e:
            print(f"Error saving user message: {e}")

    # Save file attachments to database if any
    if processed_files and db_service and user_message_id:
        for file_data in processed_files:
            if file_data.get('success'):
                try:
                    db_service.save_file_attachment(user_message_id, file_data)
                except Exception as e:
                    print(f"Error saving file attachment: {e}")

    # Convert to OpenAI format and get AI response (use llm_message with file context/image)
    # Create a temporary history with the structured message for the LLM
    temp_history = history + [{"role": "user", "content": llm_message}]
    openai_messages = messages_to_openai_format(temp_history)
    response, usage = send_message_to_llm(openai_messages, model_id, system_prompt_choice)

    # Update global token counter
    global current_conversation_tokens
    if usage and "total_tokens" in usage:
        current_conversation_tokens += usage["total_tokens"]

    # Add assistant response to history
    assistant_message = {"role": "assistant", "content": response}
    final_history = updated_history + [assistant_message]

    # Save assistant message to database
    if db_service and conversation_id:
        try:
            db_service.save_message(conversation_id, "assistant", response, model_id,
                                  system_prompt_name=system_prompt_choice)
        except Exception as e:
            print(f"Error saving assistant message: {e}")

    # Clear uploaded files and image after successful processing and return empty state
    return "", final_history, [], gr.update(visible=False, value=[]), None

def authenticate_user(username: str, password: str, remember_me: bool = False) -> Tuple[bool, str, str]:
    """Authenticate user and create session"""
    global user_service, current_user, current_conversation_id, current_conversation_tokens

    if not user_service:
        return False, "", "Authentication service not available"

    if not username or not password:
        return False, "", "Username and password are required"

    try:
        user = user_service.authenticate_user(username, password)
        if user:
            print(f"[AUTH] Authenticating user: {user.username} (ID: {user.id})")

            # Clear previous user's conversation state
            current_conversation_id = None
            current_conversation_tokens = 0
            print(f"[AUTH] Cleared conversation state for user switch")

            # Create session token
            session_token = auth_service.create_session_token(user, remember_me)

            # Set current user
            current_user = user
            print(f"[AUTH] Set current_user to: {current_user.username}")

            # Switch to user's preferences
            user_prefs = create_user_preferences(user_service, str(user.id))
            set_active_preferences(user_prefs)
            print(f"[AUTH] Loaded user preferences for user: {user.username}")

            return True, session_token, f"Welcome, {user.display_name}!"
        else:
            print(f"[AUTH] Authentication failed for username: {username}")
            return False, "", "Invalid username or password"

    except Exception as e:
        print(f"[AUTH] Authentication error for {username}: {e}")
        return False, "", "Authentication failed"

def logout_user(session_token: str) -> bool:
    """Logout user and invalidate session"""
    global current_user, current_conversation_id, current_conversation_tokens

    print(f"[AUTH] Logging out user: {current_user.username if current_user else 'Unknown'}")

    if session_token:
        auth_service.invalidate_session(session_token)

    # Clear user and conversation state
    current_user = None
    current_conversation_id = None
    current_conversation_tokens = 0
    print(f"[AUTH] Cleared user and conversation state on logout")

    # Reset to default preferences
    from user_preferences import preferences
    set_active_preferences(preferences)
    print(f"[AUTH] Reset to default preferences")

    return True

def verify_session(session_token: str) -> bool:
    """Verify if session is still valid"""
    global current_user, user_service

    if not session_token:
        print(f"[AUTH] No session token provided for verification")
        return False

    session = auth_service.get_session(session_token)
    if not session:
        print(f"[AUTH] Session not found or expired for token")
        current_user = None
        return False

    # Ensure current_user is set
    if not current_user and user_service:
        current_user = user_service.get_user_by_id(session['user_id'])
        if current_user:
            print(f"[AUTH] Restored current_user from session: {current_user.username}")

            # Also ensure user preferences are loaded for the restored user
            user_prefs = create_user_preferences(user_service, str(current_user.id))
            set_active_preferences(user_prefs)
            print(f"[AUTH] Restored user preferences for: {current_user.username}")
        else:
            print(f"[AUTH] Could not find user for session user_id: {session['user_id']}")

    is_valid = current_user is not None
    print(f"[AUTH] Session verification result: {is_valid}")
    return is_valid

def create_login_interface():
    """Create the login interface"""
    theme_map = {
        "glass": gr.themes.Glass(),
        "monochrome": gr.themes.Monochrome(),
        "soft": gr.themes.Soft(),
        "base": gr.themes.Base(),
        "default": gr.themes.Default()
    }
    theme = theme_map.get(GRADIO_THEME.lower(), gr.themes.Glass())

    with gr.Blocks(title="Local GPT - Login", theme=theme) as login_demo:
        with gr.Column(elem_id="login-container"):
            gr.HTML("<h1 style='text-align: center; margin-bottom: 2rem;'>🤖 Local GPT</h1>")
            gr.HTML("<h2 style='text-align: center; margin-bottom: 2rem;'>Please sign in to continue</h2>")

            with gr.Row():
                with gr.Column(scale=1):
                    pass  # Empty column for centering
                with gr.Column(scale=2):
                    username_input = gr.Textbox(
                        label="Username",
                        placeholder="Enter your username",
                        container=True
                    )
                    password_input = gr.Textbox(
                        label="Password",
                        placeholder="Enter your password",
                        type="password",
                        container=True
                    )
                    remember_me_checkbox = gr.Checkbox(
                        label="Remember me",
                        value=False
                    )

                    login_button = gr.Button("Sign In", variant="primary", size="lg")
                    login_message = gr.HTML("")

                with gr.Column(scale=1):
                    pass  # Empty column for centering

            # Hidden components for session management
            session_token = gr.State("")
            login_success = gr.State(False)

        def handle_login(username, password, remember_me):
            success, token, message = authenticate_user(username, password, remember_me)

            if success:
                return (
                    True,  # login_success
                    token,  # session_token
                    f"<div style='color: green; text-align: center; margin-top: 1rem;'>{message}</div>",  # login_message
                    "",  # username_input
                    ""   # password_input
                )
            else:
                return (
                    False,  # login_success
                    "",     # session_token
                    f"<div style='color: red; text-align: center; margin-top: 1rem;'>{message}</div>",  # login_message
                    username,  # keep username
                    ""      # clear password
                )

        login_button.click(
            handle_login,
            inputs=[username_input, password_input, remember_me_checkbox],
            outputs=[login_success, session_token, login_message, username_input, password_input]
        )

        # Allow Enter key to submit
        password_input.submit(
            handle_login,
            inputs=[username_input, password_input, remember_me_checkbox],
            outputs=[login_success, session_token, login_message, username_input, password_input]
        )

    return login_demo, login_success, session_token

def create_user_display_html():
    """Create HTML for user display with avatar and name"""
    global current_user

    if not current_user:
        return "<div style='text-align: right; padding: 10px;'>👤 Guest</div>"

    display_name = current_user.display_name or "User"

    # Check if user has avatar
    avatar_html = ""
    if hasattr(current_user, 'avatar_url') and current_user.avatar_url:
        import os
        if os.path.exists(current_user.avatar_url):
            avatar_html = f"<img src='file={current_user.avatar_url}' style='width: 32px; height: 32px; border-radius: 50%; margin-right: 8px; vertical-align: middle;' />"

    if not avatar_html:
        # Use emoji if no avatar
        avatar_html = "<span style='font-size: 24px; margin-right: 8px; vertical-align: middle;'>👤</span>"

    return f"""
    <div style='text-align: right; padding: 10px; display: flex; align-items: center; justify-content: flex-end;'>
        {avatar_html}
        <span style='font-weight: 500;'>{display_name}</span>
    </div>
    """

def create_interface():
    """Create the Gradio interface"""
    global user_preferences

    # Always load latest preferences
    user_preferences = load_preferences()

    initial_sidebar_visible = user_preferences.get("sidebar_visible", True)

    # Set theme based on environment variable
    theme_map = {
        "glass": gr.themes.Glass(),
        "monochrome": gr.themes.Monochrome(),
        "soft": gr.themes.Soft(),
        "base": gr.themes.Base(),
        "default": gr.themes.Default()
    }
    theme = theme_map.get(GRADIO_THEME.lower(), gr.themes.Glass())

    with gr.Blocks(title="Local GPT Chat", theme=theme) as demo:
        # State for sidebar visibility
        sidebar_visible = gr.State(initial_sidebar_visible)

        # Main layout container
        with gr.Row():
            # Left sidebar for conversations
            with gr.Column(scale=1, visible=initial_sidebar_visible, min_width=250) as sidebar:
                # Search conversations
                search_input = gr.Textbox(
                    placeholder="🔍 Search conversations...",
                    show_label=False,
                    container=False,
                    scale=1
                )

                # Conversations list
                conversations_radio = gr.Radio(
                    choices=[],
                    label="Recent Conversations",
                    interactive=True,
                    container=False
                )

            # Right main content area
            with gr.Column(scale=3) as main_content:
                # Header with model selection, user info, and sidebar toggle button
                with gr.Row():
                    sidebar_toggle_btn = gr.Button("« Hide Chat History" if initial_sidebar_visible else "» Show Chat History", variant="secondary", size="sm", scale=1)
                    with gr.Column(scale=6):
                        gr.HTML("<h1 style='text-align: center;'>🤖 Local GPT Chat</h1>")
                    with gr.Column(scale=2):
                        # User info and logout (always visible if auth enabled)
                        if AUTH_ENABLED:
                            with gr.Row() as user_controls_row:
                                user_display = gr.HTML("<div style='text-align: right; padding: 10px;'>👤 Guest</div>")
                                profile_btn = gr.Button("👤 Profile", variant="secondary", size="sm", visible=True)
                                admin_panel_btn = gr.Button("Admin Panel", variant="secondary", size="sm", visible=False)
                                logout_btn = gr.Button("Logout", variant="secondary", size="sm")
                    with gr.Column(scale=3):
                        # Model selection
                        model_choices = get_model_choices()
                        default_model = model_choices[0] if model_choices != ["No models available"] else None

                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=default_model,
                            label="Model",
                            interactive=True,
                            container=False
                        )

                        # System prompt selection - directly below model
                        # Get system prompt choices (predefined + custom)
                        def get_system_prompt_choices():
                            predefined_names = get_display_names()
                            custom_prompts = user_preferences.get("custom_system_prompts", {})
                            custom_names = [f"Custom: {name}" for name in custom_prompts.keys()]
                            return predefined_names + custom_names

                        system_prompt_choices = get_system_prompt_choices()
                        default_system_prompt = user_preferences.get("selected_system_prompt", "General Assistant")

                        system_prompt_dropdown = gr.Dropdown(
                            choices=system_prompt_choices,
                            value=default_system_prompt if default_system_prompt in system_prompt_choices else "General Assistant",
                            label="System Prompt",
                            interactive=True,
                            container=False,
                            info="Select how the AI should behave",
                            allow_custom_value=True
                        )


                        # Free models toggle - below system prompt selector
                        free_models_toggle = gr.Checkbox(
                            label="🆓 FREE",
                            value=False,
                            container=False
                        )


                # Token usage display
                with gr.Row():
                    token_usage_display = gr.HTML(
                        value="<div style='text-align: center; padding: 5px;'><span style='color: #666;'>Tokens: 0 / Unknown</span></div>",
                        visible=True
                    )

                # Main chat interface
                chatbot = gr.Chatbot(
                    type="messages",
                    value=[],
                    height=600,
                    show_label=False,
                    container=False,
                    editable="all",
                    show_copy_button=True
                )

                # Input area - message bar with image upload and send button
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Message Local GPT...",
                        show_label=True,
                        scale=5,
                        container=False,
                        lines=7,
                        max_lines=12
                    )
                    with gr.Column(scale=1, min_width=120):
                        image_input = gr.File(
                            file_types=[".png", ".jpg", ".jpeg", ".gif", ".webp"],
                            file_count="single",
                            label="📷 Image",
                            container=True,
                            height=150,
                            show_label=False,
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("Send", variant="primary", size="lg")
                        new_chat_btn = gr.Button("+ New Chat", variant="secondary", size="lg")
                        delete_chat_btn = gr.Button("- Delete", variant="stop", size="lg")
                # File upload area and buttons row
                with gr.Row():
                    with gr.Column(scale=5):
                        file_upload = gr.File(
                            file_count="multiple",
                            label="📎 Attach Files (Max 50MB each)",
                            container=True,
                            interactive=True,
                            height=150
                        )

                # File status display
                with gr.Row():
                    uploaded_files_table = gr.Dataframe(
                        headers=["📄 File", "📊 Size", "🔢 Tokens", "✅ Status"],
                        datatype=["str", "str", "str", "str"],
                        col_count=4,
                        label="",
                        visible=False,
                        interactive=False,
                        wrap=True
                    )

                # Model info in a collapsible section
                with gr.Accordion("Model Information", open=False):
                    model_info = gr.Markdown("Select a model to see its description.")

                # Custom Prompt Management
                with gr.Accordion("🎯 Custom System Prompts", open=False):
                    gr.HTML("<h4>Manage Custom System Prompts</h4>")
                    with gr.Row():
                        with gr.Column(scale=1):
                            custom_prompt_name = gr.Textbox(
                                label="Prompt Name",
                                placeholder="Enter a name for your custom prompt",
                                container=True
                            )
                            custom_prompt_text = gr.Textbox(
                                label="System Prompt",
                                placeholder="Enter your custom system prompt here...",
                                lines=6,
                                container=True,
                                info="This will be sent to the AI to define its behavior and personality"
                            )
                            with gr.Row():
                                save_custom_prompt_btn = gr.Button("💾 Save Prompt", variant="primary")
                                delete_custom_prompt_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                            custom_prompt_message = gr.HTML("")

                        with gr.Column(scale=1):
                            gr.HTML("<h5>Your Custom Prompts</h5>")
                            custom_prompts_list = gr.Dropdown(
                                choices=[],
                                label="Saved Custom Prompts",
                                interactive=True,
                                container=True,
                                info="Select a prompt to edit or delete"
                            )
                            load_custom_prompt_btn = gr.Button("📝 Load for Editing", variant="secondary")

                # User Profile Panel - visible when authenticated
                profile_panel_visible = AUTH_ENABLED
                with gr.Accordion("👤 User Profile", open=False, visible=profile_panel_visible) as profile_accordion:
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Avatar section
                            gr.HTML("<h4>Profile Picture</h4>")
                            current_avatar = gr.Image(
                                label="Current Avatar",
                                height=150,
                                width=150,
                                interactive=False,
                                show_label=False,
                                container=True
                            )
                            avatar_upload = gr.File(
                                label="📸 Upload New Avatar",
                                file_types=[".jpg", ".jpeg", ".png", ".gif", ".webp"],
                                file_count="single",
                                container=True
                            )

                        with gr.Column(scale=2):
                            # Profile information
                            gr.HTML("<h4>Profile Information</h4>")
                            profile_display_name = gr.Textbox(
                                label="Display Name",
                                placeholder="Enter your display name",
                                container=True
                            )
                            profile_user_context = gr.Textbox(
                                label="About You (for AI context)",
                                placeholder="Tell the AI about yourself for better conversations...",
                                lines=4,
                                container=True,
                                info="This information will be shared with the AI to provide more personalized responses"
                            )

                            with gr.Row():
                                save_profile_btn = gr.Button("Save Profile", variant="primary")
                                reset_profile_btn = gr.Button("Reset", variant="secondary")

                            profile_message = gr.HTML("")

                # Admin Panel - always create but only visible to admins
                admin_panel_visible = AUTH_ENABLED and current_user and current_user.is_admin
                with gr.Accordion("👑 Admin Panel", open=False, visible=admin_panel_visible) as admin_accordion:
                        gr.HTML("<h3>User Management</h3>")

                        # Create New User Section
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML("<h4>Create New User</h4>")
                                new_username = gr.Textbox(label="Username", placeholder="Enter username")
                                new_password = gr.Textbox(label="Password", placeholder="Enter password", type="password")
                                new_display_name = gr.Textbox(label="Display Name", placeholder="Enter display name (optional)")
                                new_is_admin = gr.Checkbox(label="Admin User", value=False)
                                create_user_btn = gr.Button("Create User", variant="primary")
                                create_user_message = gr.HTML("")

                            with gr.Column(scale=1):
                                gr.HTML("<h4>Existing Users</h4>")
                                users_list = gr.Dataframe(
                                    headers=["Username", "Display Name", "Admin", "Active", "Created"],
                                    datatype=["str", "str", "str", "str", "str"],
                                    label="",
                                    interactive=False,
                                    wrap=True
                                )
                                with gr.Row():
                                    refresh_users_btn = gr.Button("Refresh Users", variant="secondary")

                                # User Management Actions
                                gr.HTML("<h5>User Actions</h5>")
                                with gr.Row():
                                    target_username = gr.Textbox(label="Username", placeholder="Enter username to manage", scale=2)
                                    delete_user_btn = gr.Button("Delete User", variant="stop", scale=1)
                                user_action_message = gr.HTML("")

                        # System Settings Section
                        gr.HTML("<h4>System Settings</h4>")
                        with gr.Row():
                            auth_status = gr.HTML(f"<p>Authentication: <strong>{'Enabled' if AUTH_ENABLED else 'Disabled'}</strong></p>")
                            active_sessions = gr.HTML("")

                        refresh_status_btn = gr.Button("Refresh Status", variant="secondary")

        # State to track selected conversation ID
        selected_conversation_id = gr.State("")

        # State to track uploaded files for processing
        uploaded_files_state = gr.State([])

        # Hidden button for conversation loading
        load_conversation_btn = gr.Button("Load Conversation", visible=False)

        # Hidden trigger for conversation list refresh (triggered by authentication events)
        conversation_refresh_trigger = gr.Button("Refresh Conversations", visible=False)

        # Event handlers
        def refresh_conversations_handler():
            """Handle conversation list refresh after authentication"""
            print(f"[CONV] Conversation refresh triggered")
            return gr.update(choices=load_conversations_list(""), value=None)

        # Wire up conversation refresh trigger
        conversation_refresh_trigger.click(
            refresh_conversations_handler,
            inputs=[],
            outputs=[conversations_radio]
        )
        def handle_file_upload(files, model_choice):
            """Handle file upload and display in integrated table"""
            if not files:
                return [], gr.update(visible=False, value=[]), update_token_display(model_choice)

            global file_processor
            if not file_processor:
                error_data = [["⚠️ Error", "File processor not initialized", "0", "Failed"]]
                return [], gr.update(visible=True, value=error_data), update_token_display(model_choice)

            # Process files
            file_paths = [file.name for file in files if file and hasattr(file, 'name')]
            if not file_paths:
                error_data = [["⚠️ Error", "No valid files uploaded", "0", "Failed"]]
                return [], gr.update(visible=True, value=error_data), update_token_display(model_choice)

            processed_files = file_processor.process_multiple_files(file_paths)

            # Create table data with detailed information
            table_data = []
            total_tokens = 0
            total_size = 0
            successful_count = 0

            for file_data in processed_files:
                if file_data['file_info']:
                    filename = file_data['file_info']['filename']
                    size_bytes = file_data['file_info']['size']
                    size_display = f"{size_bytes / 1024:.1f}KB" if size_bytes < 1024*1024 else f"{size_bytes / (1024*1024):.1f}MB"

                    if file_data['success']:
                        tokens = file_data.get('token_count', 0)
                        token_display = f"{tokens:,}" if tokens > 0 else "0"
                        status = "✅ Success"
                        total_tokens += tokens
                        total_size += size_bytes
                        successful_count += 1
                    else:
                        token_display = "0"
                        error = file_data['error'][:40] + "..." if len(file_data['error']) > 40 else file_data['error']
                        status = f"ERROR: {error}"

                    table_data.append([filename, size_display, token_display, status])

            # Add summary row if multiple files
            if len(table_data) > 1:
                total_size_display = f"{total_size / 1024:.1f}KB" if total_size < 1024*1024 else f"{total_size / (1024*1024):.1f}MB"
                summary_row = [
                    f"📋 TOTAL ({successful_count}/{len(processed_files)} files)",
                    total_size_display,
                    f"{total_tokens:,}",
                    "📊 Summary"
                ]
                table_data.append(summary_row)

            # Note: File tokens are not added to conversation count here
            # They will be counted as part of API response when first message is sent

            return processed_files, gr.update(visible=True, value=table_data), update_token_display(model_choice)

        def clear_uploaded_files():
            """Clear uploaded files"""
            return [], gr.update(visible=False, value=[])

        def clear_uploaded_files_and_image():
            """Clear uploaded files and image input"""
            return [], gr.update(visible=False, value=[]), None

        def update_model_info(model_choice):
            if not model_choice or model_choice == "No models available":
                return "Select a model to see its description."

            model_name = model_choice.split("\n")[0] if "\n" in model_choice else model_choice.split(" - ")[0]

            for model in available_models:
                if model.get("name", "") == model_name:
                    description = model.get("description", "No description available.")
                    context_length = model.get("context_length", "Unknown")
                    pricing = model.get("pricing", {})

                    # Get pricing info
                    prompt_price = float(pricing.get("prompt", "0")) * 1_000_000
                    completion_price = float(pricing.get("completion", "0")) * 1_000_000
                    image_price = float(pricing.get("image", "0"))

                    price_info = "**Pricing:** "
                    price_parts = []
                    if prompt_price > 0:
                        price_parts.append(f"${prompt_price:.2f} prompt")
                    if completion_price > 0 and completion_price != prompt_price:
                        price_parts.append(f"${completion_price:.2f} completion")
                    if image_price > 0:
                        price_parts.append(f"${image_price:.3f} image")

                    context_info = f"**Context Length:** {context_length:,} tokens" if isinstance(context_length, int) else f"**Context Length:** {context_length}"

                    if price_parts:
                        price_info += ", ".join(price_parts) + " per 1M tokens"
                        return f"**{model_name}**\n\n{description}\n\n{price_info}\n\n{context_info}"
                    else:
                        return f"**{model_name}**\n\n{description}\n\n{context_info}"

            return "Model information not found."


        def handle_edit(history: List[Dict[str, str]], edit_data: gr.EditData) -> List[Dict[str, str]]:
            """Handle message editing - only updates the specific message"""
            if edit_data and edit_data.index < len(history):
                # Update the edited message content
                history[edit_data.index]["content"] = edit_data.value
                return history
            return history

        def handle_undo(history: List[Dict[str, str]], undo_data: gr.UndoData) -> Tuple[List[Dict[str, str]], str]:
            """Handle undo - remove last message and return its content to input"""
            if undo_data and undo_data.index < len(history):
                undone_message = history[undo_data.index]["content"]
                new_history = history[:undo_data.index]
                return new_history, undone_message
            return history, ""

        def handle_retry(history: List[Dict[str, str]], retry_data: gr.RetryData) -> List[Dict[str, str]]:
            """Handle retry - regenerate response from any message point"""
            global current_model_choice
            if retry_data and retry_data.index < len(history):
                # Remove messages after the retry point
                new_history = history[:retry_data.index + 1]

                # If retrying from an AI message, go back to the previous user message
                if new_history and new_history[-1]["role"] == "assistant":
                    new_history = new_history[:-1]

                # Now regenerate from the user message
                if new_history and new_history[-1]["role"] == "user":
                    model_id = extract_model_id(current_model_choice)
                    current_system_prompt = get_preference("selected_system_prompt", "General Assistant")
                    openai_messages = messages_to_openai_format(new_history)
                    response, _ = send_message_to_llm(openai_messages, model_id, current_system_prompt)

                    # Add new assistant response
                    assistant_message = {"role": "assistant", "content": response}
                    return new_history + [assistant_message]

            return history


        def get_model_context_length(model_choice: str) -> int:
            """Get context length for the selected model"""
            model_name = model_choice.split("\n")[0] if "\n" in model_choice else model_choice.split(" - ")[0]
            for model in available_models:
                if model.get("name", "") == model_name:
                    return model.get("context_length", 0)
            return 0

        def update_token_display(model_choice: str) -> str:
            """Update token usage display"""
            global current_conversation_tokens
            context_length = get_model_context_length(model_choice)

            if context_length > 0:
                percentage = min((current_conversation_tokens / context_length) * 100, 100)
                color = "#ef4444" if percentage > 90 else "#f59e0b" if percentage > 75 else "#10b981"

                return f"""<div style='text-align: center; padding: 5px;'>
                    <div style='margin-bottom: 3px;'>
                        <span style='color: {color}; font-weight: bold;'>Tokens: {current_conversation_tokens:,} / {context_length:,}</span>
                        <span style='color: #666; margin-left: 10px;'>({percentage:.1f}%)</span>
                    </div>
                    <div style='width: 100%; background-color: #e5e7eb; border-radius: 3px; height: 6px;'>
                        <div style='width: {percentage:.1f}%; background-color: {color}; height: 100%; border-radius: 3px; transition: width 0.3s ease;'></div>
                    </div>
                </div>"""
            else:
                return f"<div style='text-align: center; padding: 5px;'><span style='color: #666;'>Tokens: {current_conversation_tokens:,} / Unknown</span></div>"

        def update_current_model(model_choice):
            """Update the current model choice for retry functionality"""
            global current_model_choice
            current_model_choice = model_choice
            # Note: Model preference is now tracked automatically via conversation history
            return update_model_info(model_choice), update_token_display(model_choice)

        # Admin Panel Functions
        def create_new_user(username, password, display_name, is_admin):
            """Create a new user (admin only)"""
            global user_service, current_user

            if not current_user or not current_user.is_admin:
                return "", "ERROR: Admin access required"

            if not username or not password:
                return "", "ERROR: Username and password are required"

            try:
                user_service.create_user(
                    username=username,
                    password=password,
                    display_name=display_name or username,
                    is_admin=is_admin
                )
                return "", f"✅ User '{username}' created successfully"
            except ValueError as e:
                return "", f"ERROR: {str(e)}"
            except Exception as e:
                return "", f"ERROR: Error creating user: {str(e)}"

        def delete_user_by_username(username):
            """Delete a user by username (admin only)"""
            global user_service, current_user

            if not current_user or not current_user.is_admin:
                return "", "ERROR: Admin access required"

            if not username:
                return "", "ERROR: Username is required"

            if username == current_user.username:
                return "", "ERROR: Cannot delete your own account"

            try:
                # Find user by username
                target_user = user_service.get_user_by_username(username)
                if not target_user:
                    return "", f"ERROR: User '{username}' not found"

                # Delete the user
                success = user_service.delete_user(str(target_user.id))
                if success:
                    return "", f"✅ User '{username}' deleted successfully"
                else:
                    return "", f"ERROR: Failed to delete user '{username}'"
            except Exception as e:
                return "", f"ERROR: Error deleting user: {str(e)}"

        def load_users_list():
            """Load list of all users for admin panel"""
            global user_service, current_user

            if not current_user or not current_user.is_admin or not user_service:
                return []

            try:
                users = user_service.get_all_users()
                user_data = []
                for user in users:
                    user_data.append([
                        user.username or "N/A",
                        user.display_name or "N/A",
                        "Yes" if user.is_admin else "No",
                        "Yes" if user.is_active else "No",
                        user.created_at.strftime("%Y-%m-%d %H:%M") if user.created_at else "N/A"
                    ])
                return user_data
            except Exception as e:
                print(f"Error loading users: {e}")
                return []

        def refresh_admin_status():
            """Refresh admin panel status information"""
            global current_user

            if not current_user or not current_user.is_admin:
                return "ERROR: Admin access required", "ERROR: Admin access required"

            try:
                # Get session count
                session_count = auth_service.get_active_sessions_count()
                sessions_text = f"<p>Active Sessions: <strong>{session_count}</strong></p>"

                # Get auth status
                auth_text = f"<p>Authentication: <strong>{'Enabled' if AUTH_ENABLED else 'Disabled'}</strong></p>"

                return auth_text, sessions_text
            except Exception as e:
                error_msg = f"ERROR: Error: {str(e)}"
                return error_msg, error_msg

        def clear_user_form():
            """Clear the user creation form"""
            return "", "", "", False, ""

        # Custom Prompt Management Functions
        def refresh_system_prompt_choices():
            """Refresh system prompt dropdown choices"""
            predefined_names = get_display_names()
            custom_prompts = get_preference("custom_system_prompts", {})
            custom_names = [f"Custom: {name}" for name in custom_prompts.keys()]
            return predefined_names + custom_names

        def save_custom_prompt(prompt_name: str, prompt_text: str) -> Tuple[str, str, List[str], str]:
            """Save a custom system prompt"""
            if not prompt_name or not prompt_name.strip():
                return prompt_name, prompt_text, [], "❌ Please enter a prompt name."

            if not prompt_text or not prompt_text.strip():
                return prompt_name, prompt_text, [], "❌ Please enter the prompt text."

            prompt_name = prompt_name.strip()
            prompt_text = prompt_text.strip()

            # Get current custom prompts
            custom_prompts = get_preference("custom_system_prompts", {})

            # Save the new prompt
            custom_prompts[prompt_name] = prompt_text
            success = set_preference("custom_system_prompts", custom_prompts)

            if success:
                # Update dropdown choices
                custom_list_choices = list(custom_prompts.keys())
                # Set the newly created custom prompt as selected
                new_selected = f"Custom: {prompt_name}"
                set_preference("selected_system_prompt", new_selected)
                return "", "", custom_list_choices, f"✅ Custom prompt '{prompt_name}' saved successfully!"
            else:
                return prompt_name, prompt_text, [], "❌ Failed to save custom prompt."

        def delete_custom_prompt(selected_prompt: str) -> Tuple[List[str], List[str], str]:
            """Delete a selected custom prompt"""
            if not selected_prompt:
                return [], [], "❌ Please select a prompt to delete."

            # Get current custom prompts
            custom_prompts = get_preference("custom_system_prompts", {})

            if selected_prompt in custom_prompts:
                del custom_prompts[selected_prompt]
                success = set_preference("custom_system_prompts", custom_prompts)

                if success:
                    # Update dropdown choices
                    new_choices = refresh_system_prompt_choices()
                    custom_list_choices = list(custom_prompts.keys())
                    return new_choices, custom_list_choices, f"✅ Custom prompt '{selected_prompt}' deleted successfully!"
                else:
                    return [], [], "❌ Failed to delete custom prompt."
            else:
                return [], [], "❌ Selected prompt not found."

        def load_custom_prompt_for_editing(selected_prompt: str) -> Tuple[str, str, str]:
            """Load selected custom prompt for editing"""
            if not selected_prompt:
                return "", "", "❌ Please select a prompt to load."

            # Get current custom prompts
            custom_prompts = get_preference("custom_system_prompts", {})

            if selected_prompt in custom_prompts:
                prompt_text = custom_prompts[selected_prompt]
                return selected_prompt, prompt_text, f"📝 Loaded '{selected_prompt}' for editing."
            else:
                return "", "", "❌ Selected prompt not found."

        def update_system_prompt_selection(selected_prompt: str):
            """Update selected system prompt preference"""
            set_preference("selected_system_prompt", selected_prompt)

        # Profile Functions
        def load_user_profile():
            """Load current user's profile data"""
            global current_user
            if not current_user:
                return "", "", None, "ERROR: Not logged in"

            display_name = current_user.display_name or ""
            user_context = getattr(current_user, 'user_context', None) or ""

            # Load avatar if it exists
            avatar_path = None
            if hasattr(current_user, 'avatar_url') and current_user.avatar_url:
                import os
                if os.path.exists(current_user.avatar_url):
                    avatar_path = current_user.avatar_url

            return display_name, user_context, avatar_path, ""

        def save_user_profile(display_name, user_context):
            """Save user profile changes"""
            global current_user, user_service

            if not current_user or not user_service:
                return "ERROR: Not logged in or service unavailable"

            try:
                success = user_service.update_user_profile(
                    str(current_user.id),
                    display_name=display_name,
                    user_context=user_context
                )

                if success:
                    # Update current user object
                    current_user.display_name = display_name
                    if hasattr(current_user, 'user_context'):
                        current_user.user_context = user_context
                    return "✅ Profile updated successfully"
                else:
                    return "ERROR: Failed to update profile"

            except Exception as e:
                return f"ERROR: Error: {str(e)}"

        def handle_avatar_upload(file):
            """Handle avatar file upload"""
            global current_user, user_service, file_processor

            if not current_user or not user_service:
                return None, "ERROR: Not logged in"

            if not file:
                return None, ""

            try:
                import os
                import shutil
                from datetime import datetime

                # Create avatars directory if it doesn't exist
                avatars_dir = "uploads/avatars"
                os.makedirs(avatars_dir, exist_ok=True)

                # Generate unique filename
                file_ext = os.path.splitext(file.name)[1].lower()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"avatar_{current_user.id}_{timestamp}{file_ext}"
                avatar_path = os.path.join(avatars_dir, filename)

                # Copy uploaded file to avatar location
                shutil.copy2(file.name, avatar_path)

                # Update user's avatar URL in database
                success = user_service.update_avatar(str(current_user.id), avatar_path)

                if success:
                    # Update current user object
                    current_user.avatar_url = avatar_path
                    return avatar_path, "✅ Avatar updated successfully"
                else:
                    return None, "ERROR: Failed to save avatar"

            except Exception as e:
                return None, f"ERROR: Error uploading avatar: {str(e)}"

        def toggle_profile_panel():
            """Toggle profile panel visibility and load data"""
            profile_data = load_user_profile()
            return profile_data

        def reset_profile_form():
            """Reset profile form to current saved values"""
            return load_user_profile()


        def load_conversation_by_id(conversation_id: str, search_query: str = "") -> Tuple[List[Dict[str, str]], List[Tuple[str, str]]]:
            """Load a specific conversation by ID"""
            global current_conversation_id
            current_conversation_id = conversation_id
            messages = load_conversation_messages(conversation_id)
            return messages, load_conversations_list(search_query)

        def create_conversation_handler(conv_id: str):
            """Create a handler function for a specific conversation ID"""
            return lambda: load_conversation_by_id(conv_id)

        def handle_search(search_query: str):
            """Handle conversation search"""
            # Check if user context changed and force refresh if needed
            if check_conversations_need_refresh():
                print(f"[CONV] User context changed during search, forcing refresh")
            return gr.update(choices=load_conversations_list(search_query))

        def handle_free_models_toggle(filter_free: bool):
            """Handle toggle for filtering free models"""
            # Get filtered model choices
            new_choices = get_model_choices(filter_free=filter_free)

            # Try to preserve current selection if it's in the new list
            current_value = model_dropdown.value if hasattr(model_dropdown, 'value') else None
            new_value = current_value if current_value in new_choices else (new_choices[0] if new_choices and new_choices != ["No models available"] else None)

            return gr.update(choices=new_choices, value=new_value)

        model_dropdown.change(
            update_current_model,
            inputs=[model_dropdown],
            outputs=[model_info, token_usage_display]
        )

        # Free models toggle event handler
        free_models_toggle.change(
            handle_free_models_toggle,
            inputs=[free_models_toggle],
            outputs=[model_dropdown]
        )

        send_btn.click(
            chat_function,
            inputs=[msg_input, chatbot, model_dropdown, system_prompt_dropdown, uploaded_files_state, image_input],
            outputs=[msg_input, chatbot, uploaded_files_state, uploaded_files_table, image_input]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query)),
            inputs=[search_input],
            outputs=[conversations_radio]
        ).then(
            lambda model_choice: update_token_display(model_choice),
            inputs=[model_dropdown],
            outputs=[token_usage_display]
        )

        msg_input.submit(
            chat_function,
            inputs=[msg_input, chatbot, model_dropdown, system_prompt_dropdown, uploaded_files_state, image_input],
            outputs=[msg_input, chatbot, uploaded_files_state, uploaded_files_table, image_input]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query)),
            inputs=[search_input],
            outputs=[conversations_radio]
        ).then(
            lambda model_choice: update_token_display(model_choice),
            inputs=[model_dropdown],
            outputs=[token_usage_display]
        )


        new_chat_btn.click(
            start_new_conversation,
            inputs=[],
            outputs=[chatbot]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query), value=None),
            inputs=[search_input],
            outputs=[conversations_radio]
        ).then(
            clear_uploaded_files_and_image,
            inputs=[],
            outputs=[uploaded_files_state, uploaded_files_table, image_input]
        )

        # Conversation selection handler
        def handle_conversation_selection(selected):
            """Handle conversation selection and clear search"""
            # Check if user context changed and refresh conversations if needed
            if check_conversations_need_refresh():
                print(f"[CONV] User context changed during conversation selection, refreshing")
                # Don't select a conversation if user context changed - just refresh the list
                return [], "", [], gr.update(visible=False, value=[]), "General Assistant", None

            if selected:
                global current_conversation_id, current_conversation_tokens
                current_conversation_id = selected
                current_conversation_tokens = 0  # Reset token counter for selected conversation
                messages = load_conversation_messages(selected)

                # Get the system prompt for this conversation
                user_id = str(current_user.id) if current_user else None
                conversation_system_prompt = "General Assistant"  # Default
                if db_service and user_id:
                    try:
                        conversation_system_prompt = db_service.get_conversation_system_prompt_name(selected, user_id)
                        print(f"[CONV] Loaded system prompt for conversation {selected}: {conversation_system_prompt}")
                    except Exception as e:
                        print(f"[CONV] Error loading system prompt for conversation: {e}")

                return messages, "", [], gr.update(visible=False, value=[]), conversation_system_prompt, None  # Clear search, uploaded files, and image when selecting conversation
            return [], "", [], gr.update(visible=False, value=[]), "General Assistant", None

        conversations_radio.change(
            handle_conversation_selection,
            inputs=[conversations_radio],
            outputs=[chatbot, search_input, uploaded_files_state, uploaded_files_table, system_prompt_dropdown, image_input]
        ).then(
            lambda: gr.update(choices=load_conversations_list("")),
            outputs=[conversations_radio]
        )

        def handle_sidebar_toggle(visible):
            # Check if user context changed and refresh if needed
            if check_conversations_need_refresh():
                print(f"[CONV] User context changed during sidebar toggle, refreshing")

            new_visible = not visible
            set_preference("sidebar_visible", new_visible)
            button_text = "« Hide Chat History" if new_visible else "» Show Chat History"
            return new_visible, button_text, gr.update(visible=new_visible)


        sidebar_toggle_btn.click(
            handle_sidebar_toggle,
            inputs=[sidebar_visible],
            outputs=[sidebar_visible, sidebar_toggle_btn, sidebar]
        )


        search_input.change(
            handle_search,
            inputs=[search_input],
            outputs=[conversations_radio]
        )

        # Wire up chatbot event handlers
        chatbot.edit(
            handle_edit,
            inputs=[chatbot],
            outputs=[chatbot]
        )

        chatbot.undo(
            handle_undo,
            inputs=[chatbot],
            outputs=[chatbot, msg_input]
        )

        chatbot.retry(
            handle_retry,
            inputs=[chatbot],
            outputs=[chatbot]
        )

        # File upload event handlers
        file_upload.change(
            handle_file_upload,
            inputs=[file_upload, model_dropdown],
            outputs=[uploaded_files_state, uploaded_files_table, token_usage_display]
        )

        # Profile Panel Event Handlers
        if AUTH_ENABLED:
            profile_btn.click(
                load_user_profile,
                inputs=[],
                outputs=[profile_display_name, profile_user_context, current_avatar, profile_message]
            )

            save_profile_btn.click(
                save_user_profile,
                inputs=[profile_display_name, profile_user_context],
                outputs=[profile_message]
            ).then(
                lambda: create_user_display_html(),
                outputs=[user_display]
            )

            reset_profile_btn.click(
                reset_profile_form,
                inputs=[],
                outputs=[profile_display_name, profile_user_context, current_avatar, profile_message]
            )

            avatar_upload.change(
                handle_avatar_upload,
                inputs=[avatar_upload],
                outputs=[current_avatar, profile_message]
            ).then(
                lambda: create_user_display_html(),
                outputs=[user_display]
            )

        # Admin Panel Event Handlers - always set up but conditionally functional
        # Create user button
        create_user_btn.click(
            create_new_user,
            inputs=[new_username, new_password, new_display_name, new_is_admin],
            outputs=[new_username, create_user_message]
        ).then(
            lambda: ("", "", False),  # Clear password and checkbox
            outputs=[new_password, new_display_name, new_is_admin]
        ).then(
            load_users_list,
            outputs=[users_list]
        )

        # Refresh users button
        refresh_users_btn.click(
            load_users_list,
            outputs=[users_list]
        )

        # Refresh status button
        refresh_status_btn.click(
            refresh_admin_status,
            outputs=[auth_status, active_sessions]
        )

        # Delete user button
        delete_user_btn.click(
            delete_user_by_username,
            inputs=[target_username],
            outputs=[target_username, user_action_message]
        ).then(
            load_users_list,
            outputs=[users_list]
        )

        # System Prompt Event Handlers
        system_prompt_dropdown.change(
            update_system_prompt_selection,
            inputs=[system_prompt_dropdown],
            outputs=[]
        )

        save_custom_prompt_btn.click(
            save_custom_prompt,
            inputs=[custom_prompt_name, custom_prompt_text],
            outputs=[custom_prompt_name, custom_prompt_text, custom_prompts_list, custom_prompt_message]
        ).then(
            lambda: gr.update(choices=refresh_system_prompt_choices(), value=get_preference("selected_system_prompt", "General Assistant")),
            outputs=[system_prompt_dropdown]
        )

        delete_custom_prompt_btn.click(
            delete_custom_prompt,
            inputs=[custom_prompts_list],
            outputs=[system_prompt_dropdown, custom_prompts_list, custom_prompt_message]
        )

        load_custom_prompt_btn.click(
            load_custom_prompt_for_editing,
            inputs=[custom_prompts_list],
            outputs=[custom_prompt_name, custom_prompt_text, custom_prompt_message]
        )

        # Initialize admin panel data on load
        def initialize_admin_panel():
            users_data = load_users_list()
            auth_text, sessions_text = refresh_admin_status()
            return users_data, auth_text, sessions_text

        # Function to check if conversations need refresh due to user context change
        def check_conversations_need_refresh():
            """Check if conversation list needs refreshing due to user context change"""
            global last_conversations_user_id
            current_user_id = str(current_user.id) if current_user else None
            needs_refresh = last_conversations_user_id != current_user_id
            if needs_refresh:
                print(f"[INIT] User context changed: {last_conversations_user_id} -> {current_user_id}, refreshing conversations")
            return needs_refresh

        # Initialize conversations list and model info on startup
        def initialize_interface():
            """Initialize the interface with conversations and model info"""
            print(f"[INIT] Initializing interface, current_user: {current_user.username if current_user else 'None'}")

            # Defensive check: Don't load conversations before user authentication is established
            if AUTH_ENABLED and current_user is None:
                print(f"[INIT] Authentication enabled but no user context - deferring conversation loading")
                conversations_update = gr.update(choices=[])
            else:
                # Check if we need to refresh conversations due to user context change
                if check_conversations_need_refresh():
                    print(f"[INIT] Forcing conversation refresh due to user context change")

                # Load conversations (this will automatically use current_user context)
                conversations_update = gr.update(choices=load_conversations_list(""))

            # Update model info and token display for the default selected model
            default_model = model_dropdown.value
            if default_model and default_model != "No models available":
                model_info_text = update_model_info(default_model)
                token_display_text = update_token_display(default_model)
                # Also set the current model choice for retry functionality
                global current_model_choice
                current_model_choice = default_model

                # Initialize system prompt - use last used if authenticated user, otherwise default
                if AUTH_ENABLED and current_user and db_service:
                    try:
                        last_used_system_prompt = update_system_prompt_selection_for_user()
                        system_prompt_value = last_used_system_prompt or "General Assistant"
                        print(f"[INIT] Setting system prompt for authenticated user: {system_prompt_value}")
                    except Exception as e:
                        print(f"[INIT] Error getting last used system prompt: {e}")
                        system_prompt_value = "General Assistant"
                else:
                    system_prompt_value = get_preference("selected_system_prompt", "General Assistant")
                    print(f"[INIT] Setting system prompt from preferences: {system_prompt_value}")

                # Initialize admin panel if available
                if AUTH_ENABLED and current_user and current_user.is_admin:
                    users_data, auth_text, sessions_text = initialize_admin_panel()
                else:
                    users_data, auth_text, sessions_text = [], "", ""

                # Initialize custom prompts list
                custom_prompts = get_preference("custom_system_prompts", {})
                custom_prompts_choices = list(custom_prompts.keys())

                print(f"[INIT] Interface initialized with model and admin panel")
                return conversations_update, model_info_text, token_display_text, users_data, auth_text, sessions_text, custom_prompts_choices, system_prompt_value
            else:
                # Initialize system prompt - use last used if authenticated user, otherwise default
                if AUTH_ENABLED and current_user and db_service:
                    try:
                        last_used_system_prompt = update_system_prompt_selection_for_user()
                        system_prompt_value = last_used_system_prompt or "General Assistant"
                        print(f"[INIT] Setting system prompt for authenticated user (no model): {system_prompt_value}")
                    except Exception as e:
                        print(f"[INIT] Error getting last used system prompt (no model): {e}")
                        system_prompt_value = "General Assistant"
                else:
                    system_prompt_value = get_preference("selected_system_prompt", "General Assistant")
                    print(f"[INIT] Setting system prompt from preferences (no model): {system_prompt_value}")

                # Initialize admin panel if available
                if AUTH_ENABLED and current_user and current_user.is_admin:
                    users_data, auth_text, sessions_text = initialize_admin_panel()
                else:
                    users_data, auth_text, sessions_text = [], "", ""

                # Initialize custom prompts list
                custom_prompts = get_preference("custom_system_prompts", {})
                custom_prompts_choices = list(custom_prompts.keys())

                print(f"[INIT] Interface initialized without model")
                return (
                    conversations_update,
                    "Select a model to see its description.",
                    "<div style='text-align: center; padding: 5px;'><span style='color: #666;'>Tokens: 0 / Unknown</span></div>",
                    users_data,
                    auth_text,
                    sessions_text,
                    custom_prompts_choices,
                    system_prompt_value
                )

        # Setup demo load - always include admin panel outputs for consistency
        demo.load(
            initialize_interface,
            inputs=[],
            outputs=[conversations_radio, model_info, token_usage_display, users_list, auth_status, active_sessions, custom_prompts_list, system_prompt_dropdown]
        )

    # Return demo and user interface elements for wrapper access
    if AUTH_ENABLED:
        return demo, user_display, logout_btn, conversations_radio, model_dropdown, system_prompt_dropdown
    else:
        return demo, None, None, None, None, None

def initialize_app():
    """Initialize application globals - called both by main() and when imported for Gradio CLI"""
    global db_service, user_service, user_preferences, available_models, file_processor, AUTH_ENABLED

    if not LLM_API_KEY:
        print("ERROR: Error: LLM_API_KEY not found in environment variables")
        print("Please copy .env.example to .env and add your LLM API key")
        return False

    # Initialize database if DATABASE_URL is provided
    if os.getenv("DATABASE_URL"):
        try:
            print("Initializing database...")
            db_service = DatabaseService()

            db_service.init_db()
            print("Database initialized successfully")

            # Initialize user service if authentication is enabled
            if AUTH_ENABLED:
                print("Initializing user service...")
                user_service = UserService(db_service)

                # Initialize auth service with database
                print("Initializing auth service with database...")
                global auth_service
                auth_service = AuthService(db_service)

                # Start session cleanup background task
                start_session_cleanup_task()

                # Ensure default admin user exists
                admin_user = user_service.ensure_default_admin()
                print(f"Default admin user verified: {admin_user.username}")

        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
            print("Conversations will not be saved")
            if AUTH_ENABLED:
                print("Authentication requires database - disabling authentication")
                AUTH_ENABLED = False
    else:
        print("No DATABASE_URL found. Conversations will not be saved")
        if AUTH_ENABLED:
            print("Authentication requires database - disabling authentication")
            AUTH_ENABLED = False

    # Load user preferences (file-based if no authentication)
    user_preferences = load_preferences()
    print(f"Loaded user preferences: {len(user_preferences)} settings")

    print(f"Fetching available models from LLM API...")

    # Pre-fetch models to show any errors early
    models = fetch_available_models()
    if models:
        print(f"Found {len(models)} available models")
    else:
        print("Warning: Could not fetch models. Check your API key.")

    # Initialize file processor
    try:
        print("Initializing file processor...")
        file_processor = FileProcessor(llm_api_key=LLM_API_KEY, llm_base_url=LLM_BASE_URL)
        print("File processor initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize file processor: {e}")
        print("File upload functionality will be limited")

    return True

def start_session_cleanup_task():
    """Start background task to cleanup expired sessions"""
    def cleanup_worker():
        while True:
            try:
                if auth_service:
                    cleaned_count = auth_service.cleanup_expired_sessions()
                    if cleaned_count > 0:
                        print(f"Cleaned up {cleaned_count} expired sessions")

                # Sleep for 1 hour between cleanup runs
                time.sleep(3600)
            except Exception as e:
                print(f"Session cleanup error: {e}")
                # Sleep for 30 minutes on error before retrying
                time.sleep(1800)

    # Start cleanup thread as daemon so it stops when main app stops
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print("Session cleanup task started")

def create_app():
    """Create the main application with authentication wrapper"""
    if AUTH_ENABLED:
        # Create login interface
        login_demo, login_success, session_token = create_login_interface()

        # Create main chat interface
        chat_demo, user_display, logout_btn, conversations_radio, model_dropdown, system_prompt_dropdown = create_interface()

        # Create wrapper that shows login or chat based on authentication
        with gr.Blocks(title="Local GPT Chat") as app:
            with gr.Row():
                with gr.Column(visible=True) as login_column:
                    login_demo.render()

                with gr.Column(visible=False) as chat_column:
                    chat_demo.render()

            # Session state - use BrowserState for persistence across page refreshes
            session_state = gr.BrowserState("", storage_key="session_token")

            # Session restoration on startup
            def restore_session(stored_token):
                """Check if stored session token is still valid and restore session"""
                print(f"[DEBUG] Restoring session with token: {stored_token}")

                if stored_token and stored_token.strip():
                    print(f"[DEBUG] Verifying session token...")
                    if verify_session(stored_token):
                        # Session is valid, show chat interface
                        global current_user
                        print(f"[DEBUG] Session valid, current user: {current_user.display_name if current_user else 'None'}")
                        user_html = create_user_display_html()
                        return (
                            gr.update(visible=False),  # login_column
                            gr.update(visible=True),   # chat_column
                            stored_token,              # session_state (keep the token)
                            gr.HTML(user_html)         # user_display
                        )
                    else:
                        print(f"[DEBUG] Session invalid or expired")
                else:
                    print(f"[DEBUG] No token provided or empty token")

                # Session invalid or expired, clear it and show login
                return (
                    gr.update(visible=True),   # login_column
                    gr.update(visible=False),  # chat_column
                    "",                        # session_state (clear token)
                    gr.HTML("<div style='text-align: right; padding: 10px;'>👤 Guest</div>")  # user_display
                )

            # Handle successful login
            def show_chat_interface(success, token):
                if success and verify_session(token):
                    # Get current user and update display
                    global current_user
                    user_html = create_user_display_html()
                    return (
                        gr.update(visible=False),  # login_column
                        gr.update(visible=True),   # chat_column
                        token,                     # session_state
                        gr.HTML(user_html)         # user_display
                    )
                return (
                    gr.update(visible=True),   # login_column
                    gr.update(visible=False),  # chat_column
                    "",                        # session_state
                    gr.HTML("<div style='text-align: right; padding: 10px;'>👤 Guest</div>")  # user_display
                )

            # Connect login interface session token to wrapper session state
            def sync_session_states(success, token):
                """Sync session states and handle login success"""
                if success and token:
                    # Update both session states with the token
                    result = show_chat_interface(success, token)
                    return result
                else:
                    # Failed login - clear states
                    return (
                        gr.update(visible=True),   # login_column
                        gr.update(visible=False),  # chat_column
                        "",                        # session_state
                        gr.HTML("<div style='text-align: right; padding: 10px;'>👤 Guest</div>")  # user_display
                    )

            # Conversation refresh handler for authentication events
            def refresh_conversations_after_auth():
                """Refresh conversation list after authentication events"""
                print(f"[AUTH] Refreshing conversations after authentication event")
                # Always try to refresh - if no user is authenticated, load_conversations_list will return empty list
                return gr.update(choices=load_conversations_list(""))

            def refresh_conversations_and_model_after_auth():
                """Refresh conversation list and update model/system prompt selection after authentication events"""
                print(f"[AUTH] Refreshing conversations, model, and system prompt after authentication event")

                # Refresh conversations
                conversations_update = gr.update(choices=load_conversations_list(""))

                # Update model selection for the authenticated user
                last_used_model = update_model_selection_for_user()
                if last_used_model:
                    model_update = gr.update(value=last_used_model)
                else:
                    model_update = gr.update()  # No change if no last used model found

                # Update system prompt selection for the authenticated user
                last_used_system_prompt = update_system_prompt_selection_for_user()
                if last_used_system_prompt:
                    system_prompt_update = gr.update(value=last_used_system_prompt)
                else:
                    system_prompt_update = gr.update(value="General Assistant")  # Default fallback

                return conversations_update, model_update, system_prompt_update

            # Monitor login success and sync states
            login_success.change(
                sync_session_states,
                inputs=[login_success, session_token],
                outputs=[login_column, chat_column, session_state, user_display]
            ).then(
                refresh_conversations_and_model_after_auth,
                inputs=[],
                outputs=[conversations_radio, model_dropdown, system_prompt_dropdown]
            )

            # Logout handler at wrapper level
            def handle_logout(stored_token):
                global current_user
                if current_user:
                    logout_user(stored_token)  # Pass actual session token
                    current_user = None
                    return (
                        gr.update(visible=True),   # Show login column
                        gr.update(visible=False),  # Hide chat column
                        "",                        # Clear session state
                        gr.HTML("<div style='text-align: right; padding: 10px;'>👤 Guest</div>")  # Reset user display
                    )
                return (
                    gr.update(),  # No change to login column
                    gr.update(),  # No change to chat column
                    "",           # Clear session state
                    gr.HTML("<div style='color: red; text-align: center;'>Not logged in.</div>")
                )

            # Connect logout button to wrapper-level handler
            logout_btn.click(
                handle_logout,
                inputs=[session_state],
                outputs=[login_column, chat_column, session_state, user_display]
            )

            # Initialize interface on load and handle session restoration
            def initialize_auth_interface(stored_token):
                """Initialize authentication interface and restore session if available"""
                print(f"[DEBUG] Initializing auth interface with stored token: {stored_token}")
                if stored_token:
                    print(f"[DEBUG] Found stored token, attempting to restore session")
                    return restore_session(stored_token)
                else:
                    print(f"[DEBUG] No stored token found, showing login interface")
                    return (
                        gr.update(visible=True),   # login_column
                        gr.update(visible=False),  # chat_column
                        "",                        # session_state
                        gr.HTML("<div style='text-align: right; padding: 10px;'>👤 Guest</div>")  # user_display
                    )

            # Auto-restore session on page load
            app.load(
                initialize_auth_interface,
                inputs=[session_state],
                outputs=[login_column, chat_column, session_state, user_display]
            ).then(
                refresh_conversations_and_model_after_auth,
                inputs=[],
                outputs=[conversations_radio, model_dropdown, system_prompt_dropdown]
            )

            # Also listen for session state changes for manual restoration
            session_state.change(
                restore_session,
                inputs=[session_state],
                outputs=[login_column, chat_column, session_state, user_display]
            ).then(
                refresh_conversations_and_model_after_auth,
                inputs=[],
                outputs=[conversations_radio, model_dropdown, system_prompt_dropdown]
            )

        return app
    else:
        # No authentication - return main interface directly
        demo, _, _, _, _, _ = create_interface()
        return demo

def main():
    """Main function to run the application"""
    print("Starting Local GPT Chat...")

    if not initialize_app():
        return

    # Create and launch the interface
    if AUTH_ENABLED:
        print(f"Authentication enabled - admin login required")
        print(f"Default admin credentials: admin / admin123")

    demo = create_app()

    print(f"Starting server on http://localhost:{PORT}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True
    )

# Initialize app and create demo for Gradio CLI compatibility
initialize_app()
demo = create_app()

if __name__ == "__main__":
    main()