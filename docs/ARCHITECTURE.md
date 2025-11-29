# Architecture Overview

This document explains the technical architecture of LLM Council, including the project structure, component relationships, and data flow.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Project Structure](#project-structure)
3. [Backend Architecture](#backend-architecture)
4. [Frontend Architecture](#frontend-architecture)
5. [Data Storage](#data-storage)
6. [Communication Flow](#communication-flow)
7. [Key Design Patterns](#key-design-patterns)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER'S BROWSER                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     React Frontend (Port 5173)                        │  │
│  │                                                                       │  │
│  │   ┌─────────────┐    ┌──────────────────────────────────────────┐    │  │
│  │   │   Sidebar   │    │           ChatInterface                   │    │  │
│  │   │             │    │   ┌────────┐ ┌────────┐ ┌────────┐       │    │  │
│  │   │ - Conv List │    │   │ Stage1 │ │ Stage2 │ │ Stage3 │       │    │  │
│  │   │ - New Conv  │    │   └────────┘ └────────┘ └────────┘       │    │  │
│  │   └─────────────┘    └──────────────────────────────────────────┘    │  │
│  │                                      │                                │  │
│  │                              api.js (HTTP/SSE)                        │  │
│  └──────────────────────────────────────┼────────────────────────────────┘  │
└─────────────────────────────────────────┼───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend (Port 8001)                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           main.py                                     │  │
│  │                    (REST API Endpoints)                               │  │
│  │                                                                       │  │
│  │   GET  /api/conversations          - List all                         │  │
│  │   POST /api/conversations          - Create new                       │  │
│  │   GET  /api/conversations/{id}     - Get one                          │  │
│  │   POST /api/conversations/{id}/message        - Send (batch)          │  │
│  │   POST /api/conversations/{id}/message/stream - Send (streaming)      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                     │                           │                            │
│                     ▼                           ▼                            │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐   │
│  │      storage.py          │    │           council.py                  │   │
│  │                          │    │                                       │   │
│  │  - create_conversation() │    │  - stage1_collect_responses()        │   │
│  │  - get_conversation()    │    │  - stage2_collect_rankings()         │   │
│  │  - save_conversation()   │    │  - stage3_synthesize_final()         │   │
│  │  - list_conversations()  │    │  - parse_ranking_from_text()         │   │
│  │  - add_user_message()    │    │  - calculate_aggregate_rankings()    │   │
│  │  - add_assistant_msg()   │    │  - run_full_council()                │   │
│  └──────────────────────────┘    └──────────────────────────────────────┘   │
│            │                                      │                          │
│            ▼                                      ▼                          │
│  ┌──────────────────┐                 ┌────────────────────────────────┐    │
│  │ data/            │                 │       openrouter.py            │    │
│  │ conversations/   │                 │                                │    │
│  │   {id}.json     │                 │  - query_model()               │    │
│  │   {id}.json     │                 │  - query_models_parallel()     │    │
│  │   ...           │                 └────────────────────────────────┘    │
│  └──────────────────┘                             │                          │
└───────────────────────────────────────────────────┼──────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenRouter API                                       │
│                    (https://openrouter.ai/api/v1)                           │
│                                                                             │
│                    Routes to: OpenAI, Google, Anthropic, xAI, etc.          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
llm-council-master/
│
├── backend/                      # Python FastAPI backend
│   ├── __init__.py              # Package marker
│   ├── config.py                # Configuration and model lists
│   ├── openrouter.py            # OpenRouter API client
│   ├── council.py               # 3-stage orchestration logic
│   ├── storage.py               # JSON file persistence
│   └── main.py                  # FastAPI app and endpoints
│
├── frontend/                     # React/Vite frontend
│   ├── src/
│   │   ├── main.jsx             # React entry point
│   │   ├── App.jsx              # Main application component
│   │   ├── App.css              # App-level styles
│   │   ├── api.js               # Backend API client
│   │   ├── index.css            # Global styles
│   │   └── components/
│   │       ├── Sidebar.jsx      # Conversation list
│   │       ├── Sidebar.css
│   │       ├── ChatInterface.jsx # Main chat area
│   │       ├── ChatInterface.css
│   │       ├── Stage1.jsx       # Individual responses display
│   │       ├── Stage1.css
│   │       ├── Stage2.jsx       # Peer rankings display
│   │       ├── Stage2.css
│   │       ├── Stage3.jsx       # Final answer display
│   │       └── Stage3.css
│   ├── package.json             # Node.js dependencies
│   ├── vite.config.js           # Vite bundler config
│   └── index.html               # HTML entry point
│
├── data/                         # Created at runtime
│   └── conversations/            # JSON conversation files
│       └── {uuid}.json
│
├── docs/                         # Documentation (you are here)
│
├── .env                          # API key (not committed)
├── .gitignore                    # Git ignore rules
├── pyproject.toml               # Python dependencies
├── uv.lock                      # Python lock file
├── start.sh                     # Startup script
└── README.md                    # Project readme
```

---

## Backend Architecture

### Module Dependency Graph

```
main.py
    │
    ├── storage.py ←── config.py (DATA_DIR)
    │
    └── council.py
            │
            ├── openrouter.py ←── config.py (API_KEY, API_URL)
            │
            └── config.py (COUNCIL_MODELS, CHAIRMAN_MODEL)
```

### Module Responsibilities

#### `config.py`
**Purpose**: Central configuration

```python
# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model configuration
COUNCIL_MODELS = ["openai/gpt-5.1", "google/gemini-3-pro-preview", ...]
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Storage path
DATA_DIR = "data/conversations"
```

#### `openrouter.py`
**Purpose**: HTTP client for LLM API calls

| Function | Description |
|----------|-------------|
| `query_model(model, messages, timeout)` | Query single model, returns response or None on failure |
| `query_models_parallel(models, messages)` | Query multiple models simultaneously using asyncio |

**Key patterns**:
- Uses `httpx.AsyncClient` for async HTTP
- Graceful error handling (returns None, doesn't throw)
- 120-second default timeout

#### `council.py`
**Purpose**: 3-stage orchestration logic

| Function | Description |
|----------|-------------|
| `stage1_collect_responses(query)` | Parallel queries to all council models |
| `stage2_collect_rankings(query, stage1)` | Anonymize → query for rankings → parse |
| `stage3_synthesize_final(query, stage1, stage2)` | Chairman synthesis |
| `parse_ranking_from_text(text)` | Extract ranking list from text |
| `calculate_aggregate_rankings(stage2, mapping)` | Compute average positions |
| `generate_conversation_title(query)` | Auto-generate short title |
| `run_full_council(query)` | Execute complete 3-stage process |

#### `storage.py`
**Purpose**: JSON file persistence

| Function | Description |
|----------|-------------|
| `create_conversation(id)` | Initialize new conversation file |
| `get_conversation(id)` | Load conversation from disk |
| `save_conversation(conv)` | Write conversation to disk |
| `list_conversations()` | List all conversations (metadata only) |
| `add_user_message(id, content)` | Append user message |
| `add_assistant_message(id, s1, s2, s3)` | Append assistant response |
| `update_conversation_title(id, title)` | Update conversation title |

#### `main.py`
**Purpose**: HTTP API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/conversations` | GET | List all conversations |
| `/api/conversations` | POST | Create new conversation |
| `/api/conversations/{id}` | GET | Get specific conversation |
| `/api/conversations/{id}/message` | POST | Send message (batch) |
| `/api/conversations/{id}/message/stream` | POST | Send message (streaming) |

---

## Frontend Architecture

### Component Hierarchy

```
App.jsx (Main orchestrator)
│
├── Sidebar.jsx
│   └── Conversation list items
│
└── ChatInterface.jsx
    ├── Message display area
    │   ├── User messages
    │   └── Assistant messages
    │       ├── Stage1.jsx (tab view of responses)
    │       ├── Stage2.jsx (rankings + aggregate)
    │       └── Stage3.jsx (final answer)
    │
    └── Input form (textarea + send button)
```

### State Management

All state lives in `App.jsx`:

```javascript
// App.jsx state
const [conversations, setConversations] = useState([]);      // List of conversations
const [currentConversationId, setCurrentConversationId] = useState(null);
const [currentConversation, setCurrentConversation] = useState(null);  // Full conversation
const [isLoading, setIsLoading] = useState(false);           // Loading state
```

Data flows **down** via props:
- `App` → `Sidebar`: conversations list, current ID, handlers
- `App` → `ChatInterface`: current conversation, send handler, loading state
- `ChatInterface` → `Stage1/2/3`: stage-specific data

### API Client (`api.js`)

```javascript
const API_BASE = 'http://localhost:8001';

export const api = {
  listConversations(),      // GET /api/conversations
  createConversation(),     // POST /api/conversations
  getConversation(id),      // GET /api/conversations/{id}
  sendMessage(id, content), // POST /api/conversations/{id}/message
  sendMessageStream(id, content, onEvent),  // SSE streaming
};
```

---

## Data Storage

### Conversation JSON Structure

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00.000000",
  "title": "Learning Python Best Practices",
  "messages": [
    {
      "role": "user",
      "content": "What's the best way to learn Python?"
    },
    {
      "role": "assistant",
      "stage1": [
        {
          "model": "openai/gpt-5.1",
          "response": "Start with the official tutorial..."
        },
        {
          "model": "anthropic/claude-sonnet-4.5",
          "response": "Focus on fundamentals first..."
        }
      ],
      "stage2": [
        {
          "model": "openai/gpt-5.1",
          "ranking": "Response A provides solid advice...\n\nFINAL RANKING:\n1. Response A\n2. Response C",
          "parsed_ranking": ["Response A", "Response C", "Response B", "Response D"]
        }
      ],
      "stage3": {
        "model": "google/gemini-3-pro-preview",
        "response": "Based on the council's deliberations..."
      }
    }
  ]
}
```

### Storage Notes

- Files are stored in `data/conversations/`
- Each conversation is a separate JSON file
- File name is the conversation UUID
- **Metadata is ephemeral**: `label_to_model` and `aggregate_rankings` are NOT persisted; they're computed fresh and returned via API

---

## Communication Flow

### Streaming Flow (Primary)

```
Frontend                          Backend                         OpenRouter
   │                                 │                                 │
   │ POST /message/stream            │                                 │
   │────────────────────────────────>│                                 │
   │                                 │                                 │
   │                                 │ Save user message               │
   │                                 │ Start title generation (async) │
   │                                 │                                 │
   │ SSE: stage1_start               │                                 │
   │<────────────────────────────────│                                 │
   │                                 │ Query all models (parallel)     │
   │                                 │────────────────────────────────>│
   │                                 │<────────────────────────────────│
   │ SSE: stage1_complete {data}     │                                 │
   │<────────────────────────────────│                                 │
   │                                 │                                 │
   │ SSE: stage2_start               │                                 │
   │<────────────────────────────────│                                 │
   │                                 │ Anonymize + query rankings      │
   │                                 │────────────────────────────────>│
   │                                 │<────────────────────────────────│
   │ SSE: stage2_complete {data}     │                                 │
   │<────────────────────────────────│                                 │
   │                                 │                                 │
   │ SSE: stage3_start               │                                 │
   │<────────────────────────────────│                                 │
   │                                 │ Query chairman                  │
   │                                 │────────────────────────────────>│
   │                                 │<────────────────────────────────│
   │ SSE: stage3_complete {data}     │                                 │
   │<────────────────────────────────│                                 │
   │                                 │                                 │
   │                                 │ Wait for title (if first msg)  │
   │ SSE: title_complete {title}     │                                 │
   │<────────────────────────────────│                                 │
   │                                 │                                 │
   │                                 │ Save assistant message          │
   │                                 │                                 │
   │ SSE: complete                   │                                 │
   │<────────────────────────────────│                                 │
```

**Note**: Title generation runs in parallel with Stages 1-3 (on first message only). The `title_complete` event arrives after Stage 3 completes because that's when the backend waits for the title task to finish.

### SSE Event Types

| Event | Payload | Description |
|-------|---------|-------------|
| `stage1_start` | `{}` | Stage 1 beginning |
| `stage1_complete` | `{data: [...]}` | Stage 1 responses |
| `stage2_start` | `{}` | Stage 2 beginning |
| `stage2_complete` | `{data: [...], metadata: {...}}` | Rankings + mapping |
| `stage3_start` | `{}` | Stage 3 beginning |
| `stage3_complete` | `{data: {...}}` | Final answer |
| `title_complete` | `{data: {title: "..."}}` | Generated title |
| `complete` | `{}` | All done |
| `error` | `{message: "..."}` | Error occurred |

---

## Key Design Patterns

### 1. Async/Await Everything

The backend is fully async, enabling:
- Parallel model queries (Stage 1 and 2)
- Non-blocking I/O
- Efficient handling of long-running API calls

```python
# Example: Parallel queries
responses = await asyncio.gather(*[
    query_model(model, messages) for model in models
])
```

### 2. Graceful Degradation

System continues working even when some components fail:

```python
# In openrouter.py
async def query_model(...):
    try:
        # ... make request
    except Exception as e:
        print(f"Error: {e}")
        return None  # Continue, don't crash

# In council.py
for model, response in responses.items():
    if response is not None:  # Only include successful ones
        results.append(...)
```

### 3. Server-Sent Events (SSE)

Real-time updates without polling:

```python
# Backend yields events
async def event_generator():
    yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
    # ... do work ...
    yield f"data: {json.dumps({'type': 'stage1_complete', 'data': results})}\n\n"
```

```javascript
// Frontend processes events
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  // Parse and handle events
}
```

### 4. Optimistic UI Updates

Frontend shows immediate feedback:

```javascript
// Add user message immediately (before server confirms)
setCurrentConversation(prev => ({
  ...prev,
  messages: [...prev.messages, userMessage]
}));

// If error, remove optimistic messages
catch (error) {
  setCurrentConversation(prev => ({
    ...prev,
    messages: prev.messages.slice(0, -2)  // Remove last 2
  }));
}
```

### 5. Relative Imports

Backend modules use relative imports for proper package structure:

```python
# In council.py
from .openrouter import query_model  # Not: from openrouter import
from .config import COUNCIL_MODELS    # Not: from config import
```

This requires running as a module: `python -m backend.main`

---

## Next Steps

- **Deep dive into backend code**: [Backend Guide](./BACKEND_GUIDE.md)
- **Deep dive into frontend code**: [Frontend Guide](./FRONTEND_GUIDE.md)
- **Understand the API**: [API Reference](./API_REFERENCE.md)
