# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local ChatGPT-like interface that runs on Raspberry Pi, providing private LLM access through OpenRouter. The application uses Gradio for the web UI and is designed for self-hosted, privacy-focused AI conversations.

## Development Commands

### Running the Application
```bash
python app.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Copy `.env.example` to `.env` and configure:
- `LLM_API_KEY`: OpenRouter API key (required)
- `LLM_BASE_URL`: API endpoint (default: OpenRouter)
- `PORT`: Server port (default: 7860)

## Architecture

### Core Application Structure
- **app.py**: Single-file application containing all functionality
- **Gradio UI**: Web interface with chat, model selection, and model info
- **OpenRouter Integration**: Fetches available models and handles chat completions
- **Environment Variables**: Configuration via `.env` file

### Key Functions
- `fetch_available_models()`: Retrieves and caches model list from OpenRouter
- `send_message_to_llm()`: Handles chat API calls to selected model
- `chat_function()`: Processes user messages and manages conversation flow
- `create_interface()`: Builds the Gradio web interface

### Model Management
- Models are fetched dynamically from OpenRouter API
- Pricing information is displayed alongside model names
- Model selection affects which LLM handles subsequent messages
- No conversation history persistence (each message is independent)

### Future Architecture Notes
According to README.md, the full planned architecture includes:
- PostgreSQL with pgvector for conversation storage and semantic search
- SQLAlchemy ORM for database operations
- Redis for caching and session management
- User authentication with bcrypt password hashing
- Conversation history with vector embeddings

Currently implemented: Basic Gradio chat interface with model selection
Not yet implemented: Database storage, user authentication, conversation persistence
- pythonic, clean, minimal code
- Do not run main Gradio app - I will test it myself if you require so.