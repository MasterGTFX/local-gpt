import os
import requests
import gradio as gr
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from database import DatabaseService
from user_preferences import get_preference, set_preference, load_preferences, create_user_preferences, set_active_preferences
from file_processor import FileProcessor
from user_service import UserService
from auth import auth_service
from migrations import MigrationService

# Load environment variables
load_dotenv()

# Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
PORT = int(os.getenv("PORT", 7860))
GRADIO_THEME = os.getenv("GRADIO_THEME", "glass")

# Authentication configuration
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", 24))

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

def get_model_choices() -> List[str]:
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

    return choices

def extract_model_id(model_choice: str) -> str:
    """Extract model ID from dropdown choice"""
    # Extract model name from the first line (before newline or dash)
    model_name = model_choice.split("\n")[0] if "\n" in model_choice else model_choice.split(" - ")[0]

    for model in available_models:
        if model.get("name", "") == model_name:
            return model.get("id", "")

    # Fallback: return the model name
    return model_name

def messages_to_openai_format(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert Gradio messages format to OpenAI API format"""
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg.get("content")]

def send_message_to_llm(messages: List[Dict[str, str]], model_id: str) -> Tuple[str, Optional[Dict]]:
    """Send message to LLM API and get response with token usage"""
    try:
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Local GPT Chat"
        }

        data = {
            "model": model_id,
            "messages": messages
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
    global db_service, current_user

    if not db_service:
        return []

    try:
        user_id = str(current_user.id) if current_user else None

        if search_query and search_query.strip():
            conversations = db_service.search_conversations(search_query, limit=20, user_id=user_id)
        else:
            conversations = db_service.get_recent_conversations(limit=20, user_id=user_id)

        if not conversations:
            return []

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

        return choices

    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def load_conversation_messages(conversation_id: str) -> List[Dict[str, str]]:
    """Load conversation messages in Gradio format"""
    global db_service, current_user

    if not db_service or not conversation_id:
        return []

    try:
        # Get user ID for security verification
        user_id = str(current_user.id) if current_user else None

        # Get messages using the existing database method with user verification
        messages = db_service.get_conversation_messages(conversation_id, user_id=user_id)

        # Convert from old format [(user_msg, assistant_msg)] to new messages format
        gradio_messages = []
        for user_msg, assistant_msg in messages:
            gradio_messages.append({"role": "user", "content": user_msg})
            gradio_messages.append({"role": "assistant", "content": assistant_msg})

        return gradio_messages

    except Exception as e:
        print(f"Error loading conversation messages: {e}")
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

def chat_function(message: str, history: List[Dict[str, str]], model_choice: str, uploaded_files: List[Dict] = None) -> Tuple[str, List[Dict[str, str]], List[Dict], gr.update]:
    """Handle chat messages with optional file attachments"""
    global db_service, current_conversation_id, file_processor

    if not LLM_API_KEY:
        error_msg = {"role": "assistant", "content": "Error: Please set your LLM_API_KEY in the .env file"}
        return "", history + [error_msg], [], gr.update(visible=False, value=[])

    if not message.strip():
        return "", history, uploaded_files or [], gr.update()

    if model_choice == "No models available":
        error_msg = {"role": "assistant", "content": "Error: No models available. Please check your API key."}
        return "", history + [error_msg], [], gr.update(visible=False, value=[])

    model_id = extract_model_id(model_choice)

    # Process file attachments if any
    file_context = ""
    processed_files = uploaded_files or []
    if processed_files and file_processor:
        # Format files for LLM context
        file_context = file_processor.format_files_for_llm_context(processed_files)

    # Prepare the actual message for the LLM (combining user message with file context)
    if file_context:
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
    user_message = {"role": "user", "content": message}
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

    # Convert to OpenAI format and get AI response (use llm_message with file context)
    # Create a temporary history with the file context message for the LLM
    temp_history = history + [{"role": "user", "content": llm_message}]
    openai_messages = messages_to_openai_format(temp_history)
    response, usage = send_message_to_llm(openai_messages, model_id)

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
            db_service.save_message(conversation_id, "assistant", response, model_id)
        except Exception as e:
            print(f"Error saving assistant message: {e}")

    # Clear uploaded files after successful processing and return empty state
    return "", final_history, [], gr.update(visible=False, value=[])

def authenticate_user(username: str, password: str, remember_me: bool = False) -> Tuple[bool, str, str]:
    """Authenticate user and create session"""
    global user_service, current_user

    if not user_service:
        return False, "", "Authentication service not available"

    if not username or not password:
        return False, "", "Username and password are required"

    try:
        user = user_service.authenticate_user(username, password)
        if user:
            # Create session token
            session_token = auth_service.create_session_token(user, remember_me)

            # Set current user
            current_user = user

            # Switch to user's preferences
            user_prefs = create_user_preferences(user_service, str(user.id))
            set_active_preferences(user_prefs)

            return True, session_token, f"Welcome, {user.display_name}!"
        else:
            return False, "", "Invalid username or password"

    except Exception as e:
        print(f"Authentication error: {e}")
        return False, "", "Authentication failed"

def logout_user(session_token: str) -> bool:
    """Logout user and invalidate session"""
    global current_user

    if session_token:
        auth_service.invalidate_session(session_token)

    current_user = None

    # Reset to default preferences
    from user_preferences import preferences
    set_active_preferences(preferences)

    return True

def verify_session(session_token: str) -> bool:
    """Verify if session is still valid"""
    global current_user, user_service

    if not session_token:
        return False

    session = auth_service.get_session(session_token)
    if not session:
        current_user = None
        return False

    # Ensure current_user is set
    if not current_user and user_service:
        current_user = user_service.get_user_by_id(session['user_id'])

    return current_user is not None

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
                        label="Remember me for 30 days",
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
                        # User info and logout (only shown if authenticated)
                        if AUTH_ENABLED and current_user:
                            with gr.Row():
                                user_display = gr.HTML(f"<div style='text-align: right; padding: 10px;'>👤 {current_user.display_name}</div>")
                                if current_user.is_admin:
                                    admin_panel_btn = gr.Button("Admin Panel", variant="secondary", size="sm")
                                logout_btn = gr.Button("Logout", variant="secondary", size="sm")
                    with gr.Column(scale=3):
                        # Get saved model preference or default to first available
                        saved_model = user_preferences.get("last_selected_model")
                        model_choices = get_model_choices()
                        default_model = None

                        if saved_model and saved_model in model_choices:
                            default_model = saved_model
                        elif model_choices != ["No models available"]:
                            default_model = model_choices[0]

                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=default_model,
                            label="Model",
                            interactive=True,
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

                # Input area - full width message bar
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Message Local GPT...",
                        show_label=False,
                        scale=1,
                        container=False,
                        lines=2,
                        max_lines=4
                    )

                # File upload area and buttons row
                with gr.Row():
                    with gr.Column(scale=8):
                        file_upload = gr.File(
                            file_count="multiple",
                            label="📎 Attach Files (Max 50MB each)",
                            file_types=[
                                ".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls",
                                ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
                                ".mp3", ".wav", ".m4a", ".ogg", ".flac",
                                ".txt", ".md", ".html", ".htm", ".csv", ".json", ".xml", ".yaml", ".yml",
                                ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".css", ".sql"
                            ],
                            container=True,
                            interactive=True
                        )
                    with gr.Column(scale=2):
                        send_btn = gr.Button("Send", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear", variant="secondary", size="lg")
                        new_chat_btn = gr.Button("+ New Chat", variant="stop", size="lg")

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

        # Event handlers
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
                        status = f"❌ {error}"

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

        def clear_conversation():
            """Clear the conversation history and start a new conversation"""
            global current_conversation_id, current_conversation_tokens
            current_conversation_id = None  # This will trigger creation of new conversation on next message
            current_conversation_tokens = 0
            return []

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
                    openai_messages = messages_to_openai_format(new_history)
                    response, _ = send_message_to_llm(openai_messages, model_id)

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
            # Save model preference
            set_preference("last_selected_model", model_choice)
            return update_model_info(model_choice), update_token_display(model_choice)

        # Admin Panel Functions
        def create_new_user(username, password, display_name, is_admin):
            """Create a new user (admin only)"""
            global user_service, current_user

            if not current_user or not current_user.is_admin:
                return "", "❌ Admin access required"

            if not username or not password:
                return "", "❌ Username and password are required"

            try:
                user_service.create_user(
                    username=username,
                    password=password,
                    display_name=display_name or username,
                    is_admin=is_admin
                )
                return "", f"✅ User '{username}' created successfully"
            except ValueError as e:
                return "", f"❌ {str(e)}"
            except Exception as e:
                return "", f"❌ Error creating user: {str(e)}"

        def delete_user_by_username(username):
            """Delete a user by username (admin only)"""
            global user_service, current_user

            if not current_user or not current_user.is_admin:
                return "", "❌ Admin access required"

            if not username:
                return "", "❌ Username is required"

            if username == current_user.username:
                return "", "❌ Cannot delete your own account"

            try:
                # Find user by username
                target_user = user_service.get_user_by_username(username)
                if not target_user:
                    return "", f"❌ User '{username}' not found"

                # Delete the user
                success = user_service.delete_user(str(target_user.id))
                if success:
                    return "", f"✅ User '{username}' deleted successfully"
                else:
                    return "", f"❌ Failed to delete user '{username}'"
            except Exception as e:
                return "", f"❌ Error deleting user: {str(e)}"

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
                return "❌ Admin access required", "❌ Admin access required"

            try:
                # Get session count
                session_count = auth_service.get_active_sessions_count()
                sessions_text = f"<p>Active Sessions: <strong>{session_count}</strong></p>"

                # Get auth status
                auth_text = f"<p>Authentication: <strong>{'Enabled' if AUTH_ENABLED else 'Disabled'}</strong></p>"

                return auth_text, sessions_text
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return error_msg, error_msg

        def clear_user_form():
            """Clear the user creation form"""
            return "", "", "", False, ""

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
            return gr.update(choices=load_conversations_list(search_query))

        model_dropdown.change(
            update_current_model,
            inputs=[model_dropdown],
            outputs=[model_info, token_usage_display]
        )

        send_btn.click(
            chat_function,
            inputs=[msg_input, chatbot, model_dropdown, uploaded_files_state],
            outputs=[msg_input, chatbot, uploaded_files_state, uploaded_files_table]
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
            inputs=[msg_input, chatbot, model_dropdown, uploaded_files_state],
            outputs=[msg_input, chatbot, uploaded_files_state, uploaded_files_table]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query)),
            inputs=[search_input],
            outputs=[conversations_radio]
        ).then(
            lambda model_choice: update_token_display(model_choice),
            inputs=[model_dropdown],
            outputs=[token_usage_display]
        )

        clear_btn.click(
            clear_conversation,
            inputs=[],
            outputs=[chatbot]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query), value=None),
            inputs=[search_input],
            outputs=[conversations_radio]
        ).then(
            clear_uploaded_files,
            inputs=[],
            outputs=[uploaded_files_state, uploaded_files_table]
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
            clear_uploaded_files,
            inputs=[],
            outputs=[uploaded_files_state, uploaded_files_table]
        )

        # Conversation selection handler
        def handle_conversation_selection(selected):
            """Handle conversation selection and clear search"""
            if selected:
                global current_conversation_id, current_conversation_tokens
                current_conversation_id = selected
                current_conversation_tokens = 0  # Reset token counter for selected conversation
                messages = load_conversation_messages(selected)
                return messages, "", [], gr.update(visible=False, value=[])  # Clear search and uploaded files when selecting conversation
            return [], "", [], gr.update(visible=False, value=[])

        conversations_radio.change(
            handle_conversation_selection,
            inputs=[conversations_radio],
            outputs=[chatbot, search_input, uploaded_files_state, uploaded_files_table]
        ).then(
            lambda: gr.update(choices=load_conversations_list("")),
            outputs=[conversations_radio]
        )

        def handle_sidebar_toggle(visible):
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

        # Logout button handler (only set up if auth is enabled and user is logged in)
        if AUTH_ENABLED and current_user:
            def handle_logout():
                logout_user("")  # Clear session
                return gr.HTML("<div style='color: green; text-align: center;'>Logged out successfully. Refresh the page to login again.</div>")

            logout_btn.click(
                handle_logout,
                outputs=[user_display]
            )

        # Initialize admin panel data on load
        def initialize_admin_panel():
            users_data = load_users_list()
            auth_text, sessions_text = refresh_admin_status()
            return users_data, auth_text, sessions_text

        # Initialize conversations list and model info on startup
        def initialize_interface():
            """Initialize the interface with conversations and model info"""
            # Load conversations
            conversations_update = gr.update(choices=load_conversations_list(""))

            # Update model info and token display for the default selected model
            default_model = model_dropdown.value
            if default_model and default_model != "No models available":
                model_info_text = update_model_info(default_model)
                token_display_text = update_token_display(default_model)
                # Also set the current model choice for retry functionality
                global current_model_choice
                current_model_choice = default_model

                # Initialize admin panel if available
                if AUTH_ENABLED and current_user and current_user.is_admin:
                    users_data, auth_text, sessions_text = initialize_admin_panel()
                else:
                    users_data, auth_text, sessions_text = [], "", ""

                return conversations_update, model_info_text, token_display_text, users_data, auth_text, sessions_text
            else:
                # Initialize admin panel if available
                if AUTH_ENABLED and current_user and current_user.is_admin:
                    users_data, auth_text, sessions_text = initialize_admin_panel()
                else:
                    users_data, auth_text, sessions_text = [], "", ""

                return (
                    conversations_update,
                    "Select a model to see its description.",
                    "<div style='text-align: center; padding: 5px;'><span style='color: #666;'>Tokens: 0 / Unknown</span></div>",
                    users_data,
                    auth_text,
                    sessions_text
                )

        # Setup demo load - always include admin panel outputs for consistency
        demo.load(
            initialize_interface,
            inputs=[],
            outputs=[conversations_radio, model_info, token_usage_display, users_list, auth_status, active_sessions]
        )

    return demo

def initialize_app():
    """Initialize application globals - called both by main() and when imported for Gradio CLI"""
    global db_service, user_service, user_preferences, available_models, file_processor, AUTH_ENABLED

    if not LLM_API_KEY:
        print("❌ Error: LLM_API_KEY not found in environment variables")
        print("Please copy .env.example to .env and add your LLM API key")
        return False

    # Initialize database if DATABASE_URL is provided
    if os.getenv("DATABASE_URL"):
        try:
            print("Initializing database...")
            db_service = DatabaseService()

            # Run migrations if authentication is enabled
            if AUTH_ENABLED:
                print("Authentication enabled - checking for migrations...")
                migration_service = MigrationService()
                if migration_service.check_migration_needed():
                    print("Running database migrations...")
                    if not migration_service.run_migrations():
                        print("❌ Database migration failed")
                        return False
                else:
                    print("Database is up to date")

            db_service.init_db()
            print("Database initialized successfully")

            # Initialize user service if authentication is enabled
            if AUTH_ENABLED:
                print("Initializing user service...")
                user_service = UserService(db_service)

                # Ensure default admin user exists
                admin_user = user_service.ensure_default_admin()
                print(f"Default admin user verified: {admin_user.username}")

        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
            print("Conversations will not be saved")
            if AUTH_ENABLED:
                print("❌ Authentication requires database - disabling authentication")
                AUTH_ENABLED = False
    else:
        print("No DATABASE_URL found. Conversations will not be saved")
        if AUTH_ENABLED:
            print("❌ Authentication requires database - disabling authentication")
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

def create_app():
    """Create the main application with authentication wrapper"""
    if AUTH_ENABLED:
        # Create login interface
        login_demo, login_success, session_token = create_login_interface()

        # Create main chat interface
        chat_demo = create_interface()

        # Create wrapper that shows login or chat based on authentication
        with gr.Blocks(title="Local GPT Chat") as app:
            with gr.Row():
                with gr.Column(visible=True) as login_column:
                    login_demo.render()

                with gr.Column(visible=False) as chat_column:
                    chat_demo.render()

            # Session state
            session_state = gr.State("")

            # Handle successful login
            def show_chat_interface(success, token):
                if success and verify_session(token):
                    return (
                        gr.update(visible=False),  # login_column
                        gr.update(visible=True),   # chat_column
                        token                      # session_state
                    )
                return (
                    gr.update(visible=True),   # login_column
                    gr.update(visible=False),  # chat_column
                    ""                         # session_state
                )

            # Monitor login success
            login_success.change(
                show_chat_interface,
                inputs=[login_success, session_token],
                outputs=[login_column, chat_column, session_state]
            )

        return app
    else:
        # No authentication - return main interface directly
        return create_interface()

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