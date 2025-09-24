import os
import requests
import gradio as gr
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from database import DatabaseService
from user_preferences import get_preference, set_preference, load_preferences

# Load environment variables
load_dotenv()

# Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
PORT = int(os.getenv("PORT", 7860))

# Global variables
available_models = []
selected_model = None
db_service: Optional[DatabaseService] = None
current_conversation_id: Optional[str] = None
current_model_choice = ""
user_preferences = {}

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

def send_message_to_llm(messages: List[Dict[str, str]], model_id: str) -> str:
    """Send message to LLM API and get response"""
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
            return result["choices"][0]["message"]["content"]
        else:
            return "Error: No response from model"

    except Exception as e:
        return f"Error: {str(e)}"

def ensure_conversation() -> str:
    """Ensure we have a current conversation, create one if needed"""
    global current_conversation_id, db_service

    if not current_conversation_id and db_service:
        current_conversation_id = db_service.create_conversation()

    return current_conversation_id

def generate_conversation_title(content: str) -> str:
    """Generate conversation title from first 3 words of user message"""
    if not content or not content.strip():
        return "New Conversation"

    words = content.strip().split()[:3]
    return " ".join(words) if words else "New Conversation"

def load_conversations_list(search_query: str = "") -> List[Tuple[str, str]]:
    """Load conversations for Radio component"""
    global db_service

    if not db_service:
        return []

    try:
        if search_query and search_query.strip():
            conversations = db_service.search_conversations(search_query, limit=20)
        else:
            conversations = db_service.get_recent_conversations(limit=20)

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
    global db_service

    if not db_service or not conversation_id:
        return []

    try:
        # Get messages using the existing database method
        messages = db_service.get_conversation_messages(conversation_id)

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
    global current_conversation_id
    current_conversation_id = None  # Will be created when first message is sent
    return []

def chat_function(message: str, history: List[Dict[str, str]], model_choice: str) -> Tuple[str, List[Dict[str, str]]]:
    """Handle chat messages"""
    global db_service, current_conversation_id

    if not LLM_API_KEY:
        error_msg = {"role": "assistant", "content": "Error: Please set your LLM_API_KEY in the .env file"}
        return "", history + [error_msg]

    if not message.strip():
        return "", history

    if model_choice == "No models available":
        error_msg = {"role": "assistant", "content": "Error: No models available. Please check your API key."}
        return "", history + [error_msg]

    model_id = extract_model_id(model_choice)

    # Ensure we have a conversation to save to
    conversation_id = ensure_conversation()

    # If this is the first message in a new conversation, set the title
    if db_service and conversation_id and len(history) == 0:
        title = generate_conversation_title(message)
        try:
            db_service.set_conversation_title(conversation_id, title)
        except Exception as e:
            print(f"Error setting conversation title: {e}")

    # Add user message to history
    user_message = {"role": "user", "content": message}
    updated_history = history + [user_message]

    # Save user message to database
    if db_service and conversation_id:
        try:
            db_service.save_message(conversation_id, "user", message, model_id)
        except Exception as e:
            print(f"Error saving user message: {e}")

    # Convert to OpenAI format and get AI response
    openai_messages = messages_to_openai_format(updated_history)
    response = send_message_to_llm(openai_messages, model_id)

    # Add assistant response to history
    assistant_message = {"role": "assistant", "content": response}
    final_history = updated_history + [assistant_message]

    # Save assistant message to database
    if db_service and conversation_id:
        try:
            db_service.save_message(conversation_id, "assistant", response, model_id)
        except Exception as e:
            print(f"Error saving assistant message: {e}")

    return "", final_history

def create_interface():
    """Create the Gradio interface"""
    global user_preferences

    # Always load latest preferences
    user_preferences = load_preferences()

    initial_sidebar_visible = user_preferences.get("sidebar_visible", True)
    theme_preference = user_preferences.get("theme_preference", "light")

    # Set theme based on preference
    if theme_preference == "dark":
        theme = gr.themes.Monochrome()
    else:
        theme = gr.themes.Glass()

    with gr.Blocks(title="Local GPT Chat", theme=theme) as demo:
        # State for sidebar visibility and theme
        sidebar_visible = gr.State(initial_sidebar_visible)
        current_theme = gr.State(theme_preference)

        # Main layout container
        with gr.Row():
            # Left sidebar for conversations
            with gr.Column(scale=1, visible=initial_sidebar_visible, min_width=250) as sidebar:
                # New Chat button
                new_chat_btn = gr.Button("+ New Chat", variant="primary", size="lg")

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
                # Header with model selection, theme toggle, and sidebar toggle button
                with gr.Row():
                    sidebar_toggle_btn = gr.Button("« Hide Chat History" if initial_sidebar_visible else "» Show Chat History", variant="secondary", size="sm", scale=1)
                    theme_toggle_btn = gr.Button("🌙" if theme_preference == "light" else "☀️", variant="secondary", size="sm", scale=1)
                    with gr.Column(scale=8):
                        gr.HTML("<h1 style='text-align: center;'>🤖 Local GPT Chat</h1>")
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

                # Input area at bottom
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Message Local GPT...",
                        show_label=False,
                        scale=8,
                        container=False
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")
                    clear_btn = gr.Button("Clear", scale=1, variant="secondary")

                # Model info in a collapsible section
                with gr.Accordion("Model Information", open=False):
                    model_info = gr.Markdown("Select a model to see its description.")

        # State to track selected conversation ID
        selected_conversation_id = gr.State("")

        # Hidden button for conversation loading
        load_conversation_btn = gr.Button("Load Conversation", visible=False)

        # Event handlers
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
            global current_conversation_id
            current_conversation_id = None  # This will trigger creation of new conversation on next message
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
                    response = send_message_to_llm(openai_messages, model_id)

                    # Add new assistant response
                    assistant_message = {"role": "assistant", "content": response}
                    return new_history + [assistant_message]

            return history


        def update_current_model(model_choice):
            """Update the current model choice for retry functionality"""
            global current_model_choice
            current_model_choice = model_choice
            # Save model preference
            set_preference("last_selected_model", model_choice)
            return update_model_info(model_choice)

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
            outputs=[model_info]
        )

        send_btn.click(
            chat_function,
            inputs=[msg_input, chatbot, model_dropdown],
            outputs=[msg_input, chatbot]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query)),
            inputs=[search_input],
            outputs=[conversations_radio]
        )

        msg_input.submit(
            chat_function,
            inputs=[msg_input, chatbot, model_dropdown],
            outputs=[msg_input, chatbot]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query)),
            inputs=[search_input],
            outputs=[conversations_radio]
        )

        clear_btn.click(
            clear_conversation,
            inputs=[],
            outputs=[chatbot]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query), value=None),
            inputs=[search_input],
            outputs=[conversations_radio]
        )

        new_chat_btn.click(
            start_new_conversation,
            inputs=[],
            outputs=[chatbot]
        ).then(
            lambda search_query: gr.update(choices=load_conversations_list(search_query), value=None),
            inputs=[search_input],
            outputs=[conversations_radio]
        )

        # Conversation selection handler
        def handle_conversation_selection(selected):
            """Handle conversation selection and clear search"""
            if selected:
                global current_conversation_id
                current_conversation_id = selected
                messages = load_conversation_messages(selected)
                return messages, ""  # Clear search when selecting conversation
            return [], ""

        conversations_radio.change(
            handle_conversation_selection,
            inputs=[conversations_radio],
            outputs=[chatbot, search_input]
        ).then(
            lambda: gr.update(choices=load_conversations_list("")),
            outputs=[conversations_radio]
        )

        def handle_sidebar_toggle(visible):
            new_visible = not visible
            set_preference("sidebar_visible", new_visible)
            button_text = "« Hide Chat History" if new_visible else "» Show Chat History"
            return new_visible, button_text, gr.update(visible=new_visible)

        def handle_theme_toggle(current_theme_val):
            """Toggle between light and dark theme"""
            new_theme = "dark" if current_theme_val == "light" else "light"
            set_preference("theme_preference", new_theme)
            new_button_text = "🌙" if new_theme == "light" else "☀️"

            # Show meaningful notification about theme change
            theme_name = "dark" if new_theme == "dark" else "light"
            gr.Info(f"Switched to {theme_name} theme! Please refresh the page to see the change.")

            return new_theme, new_button_text

        sidebar_toggle_btn.click(
            handle_sidebar_toggle,
            inputs=[sidebar_visible],
            outputs=[sidebar_visible, sidebar_toggle_btn, sidebar]
        )

        theme_toggle_btn.click(
            handle_theme_toggle,
            inputs=[current_theme],
            outputs=[current_theme, theme_toggle_btn]
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

        # Initialize conversations list on startup
        demo.load(
            lambda: gr.update(choices=load_conversations_list("")),
            inputs=[],
            outputs=[conversations_radio]
        )

    return demo

def main():
    """Main function to run the application"""
    global db_service

    if not LLM_API_KEY:
        print("❌ Error: LLM_API_KEY not found in environment variables")
        print("Please copy .env.example to .env and add your LLM API key")
        return

    print("Starting Local GPT Chat...")

    # Load user preferences
    global user_preferences
    user_preferences = load_preferences()
    print(f"Loaded user preferences: {len(user_preferences)} settings")

    # Initialize database if DATABASE_URL is provided
    if os.getenv("DATABASE_URL"):
        try:
            print("Initializing database...")
            db_service = DatabaseService()
            db_service.init_db()
            print("Database initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize database: {e}")
            print("Conversations will not be saved")
    else:
        print("No DATABASE_URL found. Conversations will not be saved")

    print(f"Fetching available models from LLM API...")

    # Pre-fetch models to show any errors early
    models = fetch_available_models()
    if models:
        print(f"Found {len(models)} available models")
    else:
        print("Warning: Could not fetch models. Check your API key.")

    # Create and launch the interface
    demo = create_interface()

    print(f"Starting server on http://localhost:{PORT}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()