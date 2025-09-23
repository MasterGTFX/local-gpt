# Local LLM Chat Interface

A private, self-hosted ChatGPT-like interface running on Raspberry Pi. This application provides a secure way to interact with various LLM models through OpenRouter while keeping your conversation history stored locally.

## Overview

A local web interface for chatting with Large Language Models similar to ChatGPT, with complete control over your data and privacy. Uses OpenRouter to access multiple AI models (Claude, GPT-4, Llama, etc.) while storing conversations locally in PostgreSQL with vector embeddings for semantic search.

## Key Features

- **Private & Local**: All data stays on your Raspberry Pi
- **Multiple Models**: Access various LLMs through OpenRouter
- **User Profiles**: Individual settings and conversation histories
- **Conversation Memory**: PostgreSQL with vector embeddings
- **Smart Caching**: Redis caches model lists and frequent queries
- **Semantic Search**: Find past conversations easily
- **Web Interface**: Clean Gradio-based UI

## Technology Stack

### Backend
- **Python 3.9+**: Core application
- **Gradio**: Web UI framework
- **SQLAlchemy**: Database ORM
- **OpenRouter**: Multi-model LLM access
- **Redis**: Caching layer

### Database
- **PostgreSQL + pgvector**: Conversations with semantic search
- **Redis**: Cache for model lists, rate limits, sessions

### Security
- **Passlib + bcrypt**: Password hashing
- **python-dotenv**: Environment variables

## Architecture

```
Browser ──▶ Gradio UI ──▶ Python Backend ──▶ OpenRouter API
                              │
                              ├──▶ PostgreSQL (conversations + vectors)
                              └──▶ Redis (cache models, sessions)
```

## Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `password_hash`: Bcrypt hashed password
- `created_at`: Account creation timestamp
- `settings`: JSON field for user preferences

### Conversations Table
- `id`: Primary key
- `user_id`: Foreign key to users
- `title`: Conversation title (auto-generated)
- `created_at`: Timestamp
- `updated_at`: Last message timestamp

### Messages Table
- `id`: Primary key
- `conversation_id`: Foreign key to conversations
- `role`: 'user' or 'assistant'
- `content`: Message text
- `embedding`: Vector embedding (pgvector)
- `model`: Model used for response
- `timestamp`: Message timestamp

## Quick Start

1. Clone repository and install dependencies
2. Set up PostgreSQL with pgvector extension
3. Configure `.env` file with API keys
4. Run `python app.py`
5. Access at `http://localhost:7860`

## Configuration

Environment variables (`.env`):
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection (default: localhost:6379)
- `SECRET_KEY`: For session management

## Redis Caching Strategy

- **Available models**: Cached for 1 hour (reduces API calls)
- **Model pricing**: Cached for 24 hours
- **User sessions**: Stored in Redis for fast access
- **Rate limiting**: Track API usage per user

## Usage

Access at `http://raspberry-pi-ip:7860` from any device on your network. Login, select a model from the cached list, and start chatting. All conversations are saved and searchable.

## Future Ideas

- RAG using conversation history
- Export conversations
- Model comparison
- Custom system prompts
- Voice input/output

## License

MIT License - Use freely for personal and commercial projects

## Privacy

Complete privacy - no data sent to third parties except LLM API calls to OpenRouter. All conversation history stays local.
