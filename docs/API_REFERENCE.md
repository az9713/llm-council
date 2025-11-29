# API Reference

Complete documentation for the LLM Council REST API.

## Base URL

```
http://localhost:8001
```

## Authentication

No authentication required for local development.

---

## Endpoints

### Health Check

Check if the API is running.

```
GET /
```

**Response**
```json
{
  "status": "ok",
  "service": "LLM Council API"
}
```

---

### List Conversations

Get metadata for all conversations, optionally filtered by tag.

```
GET /api/conversations
GET /api/conversations?tag=coding
```

**Query Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `tag` | string | No | Filter conversations to only those with this tag |

**Response**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2024-01-15T10:30:00.000000",
    "title": "Learning Python",
    "message_count": 4,
    "tags": ["coding", "learning"]
  },
  {
    "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "created_at": "2024-01-14T09:15:00.000000",
    "title": "REST vs GraphQL",
    "message_count": 2,
    "tags": ["technical", "research"]
  }
]
```

**Notes**
- Returns newest conversations first
- Only metadata is returned, not full message content
- `message_count` includes both user and assistant messages
- `tags` is an array of strings (may be empty)

---

### Create Conversation

Create a new empty conversation.

```
POST /api/conversations
```

**Request Body**
```json
{}
```

**Response**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00.000000",
  "title": "New Conversation",
  "tags": [],
  "messages": []
}
```

**Notes**
- ID is generated as a UUID v4
- Title starts as "New Conversation" and is updated after first message
- Tags start as an empty array

---

### Get Conversation

Get a specific conversation with all messages.

```
GET /api/conversations/{conversation_id}
```

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `conversation_id` | string | UUID of the conversation |

**Response**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00.000000",
  "title": "Learning Python",
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
          "response": "Start with the official Python tutorial..."
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

**Error Responses**
| Status | Description |
|--------|-------------|
| 404 | Conversation not found |

---

### Send Message (Batch)

Send a message and receive all stages at once.

```
POST /api/conversations/{conversation_id}/message
```

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `conversation_id` | string | UUID of the conversation |

**Request Body**
```json
{
  "content": "What's the best way to learn Python?",
  "system_prompt": "You are a helpful coding tutor."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | The user's question |
| `system_prompt` | string | No | Optional system prompt prepended to all model queries |

**Response**
```json
{
  "stage1": [
    {
      "model": "openai/gpt-5.1",
      "response": "Start with the official Python tutorial..."
    },
    {
      "model": "anthropic/claude-sonnet-4.5",
      "response": "Focus on fundamentals first..."
    },
    {
      "model": "google/gemini-3-pro-preview",
      "response": "I recommend a structured approach..."
    },
    {
      "model": "x-ai/grok-4",
      "response": "Dive into coding immediately..."
    }
  ],
  "stage2": [
    {
      "model": "openai/gpt-5.1",
      "ranking": "Detailed evaluation text...\n\nFINAL RANKING:\n1. Response C\n2. Response A\n3. Response B\n4. Response D",
      "parsed_ranking": ["Response C", "Response A", "Response B", "Response D"]
    },
    {
      "model": "anthropic/claude-sonnet-4.5",
      "ranking": "Another evaluation...\n\nFINAL RANKING:\n1. Response A\n2. Response C\n3. Response B\n4. Response D",
      "parsed_ranking": ["Response A", "Response C", "Response B", "Response D"]
    }
  ],
  "stage3": {
    "model": "google/gemini-3-pro-preview",
    "response": "Based on the council's deliberations, here is the synthesized answer..."
  },
  "metadata": {
    "label_to_model": {
      "Response A": "openai/gpt-5.1",
      "Response B": "anthropic/claude-sonnet-4.5",
      "Response C": "google/gemini-3-pro-preview",
      "Response D": "x-ai/grok-4"
    },
    "aggregate_rankings": [
      {
        "model": "google/gemini-3-pro-preview",
        "average_rank": 1.5,
        "rankings_count": 4
      },
      {
        "model": "openai/gpt-5.1",
        "average_rank": 2.0,
        "rankings_count": 4
      },
      {
        "model": "anthropic/claude-sonnet-4.5",
        "average_rank": 2.75,
        "rankings_count": 4
      },
      {
        "model": "x-ai/grok-4",
        "average_rank": 3.75,
        "rankings_count": 4
      }
    ]
  }
}
```

**Error Responses**
| Status | Description |
|--------|-------------|
| 404 | Conversation not found |

**Notes**
- This endpoint waits until all 3 stages complete before returning
- For real-time updates, use the streaming endpoint instead
- The conversation is automatically saved after completion

---

### Send Message (Streaming)

Send a message and receive Server-Sent Events as each stage completes.

```
POST /api/conversations/{conversation_id}/message/stream
```

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `conversation_id` | string | UUID of the conversation |

**Request Body**
```json
{
  "content": "What's the best way to learn Python?",
  "system_prompt": "You are a helpful coding tutor."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | The user's question |
| `system_prompt` | string | No | Optional system prompt prepended to all model queries |

**Response**

Content-Type: `text/event-stream`

Events are sent in this order:

#### 1. Stage 1 Start
```
data: {"type": "stage1_start"}
```

#### 2. Stage 1 Complete
```
data: {"type": "stage1_complete", "data": [{"model": "openai/gpt-5.1", "response": "..."}, ...]}
```

#### 3. Stage 2 Start
```
data: {"type": "stage2_start"}
```

#### 4. Stage 2 Complete
```
data: {"type": "stage2_complete", "data": [...], "metadata": {"label_to_model": {...}, "aggregate_rankings": [...]}}
```

#### 5. Stage 3 Start
```
data: {"type": "stage3_start"}
```

#### 6. Stage 3 Complete
```
data: {"type": "stage3_complete", "data": {"model": "...", "response": "..."}}
```

#### 7. Title Complete (first message only)
```
data: {"type": "title_complete", "data": {"title": "Learning Python"}}
```

#### 8. Complete
```
data: {"type": "complete"}
```

#### Error (if something goes wrong)
```
data: {"type": "error", "message": "Error description"}
```

---

### Update Conversation Tags

Update tags for a specific conversation.

```
PUT /api/conversations/{conversation_id}/tags
```

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| `conversation_id` | string | UUID of the conversation |

**Request Body**
```json
{
  "tags": ["coding", "learning", "python"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tags` | array | Yes | Array of tag strings |

**Response**
```json
{
  "status": "ok",
  "tags": ["coding", "learning", "python"]
}
```

**Error Responses**
| Status | Description |
|--------|-------------|
| 404 | Conversation not found |

**Notes**
- Tags are stored as lowercase strings
- Replaces all existing tags with the provided array
- Pass an empty array to remove all tags

---

### Get All Tags

Get all unique tags used across all conversations.

```
GET /api/tags
```

**Response**
```json
{
  "tags": ["analysis", "coding", "creative", "learning", "research", "technical"]
}
```

**Notes**
- Returns tags sorted alphabetically
- Only includes tags that are currently used by at least one conversation
- Useful for populating tag filter dropdowns

---

### Get Configuration

Get current model configuration.

```
GET /api/config
```

**Response**
```json
{
  "council_models": [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4"
  ],
  "chairman_model": "google/gemini-3-pro-preview"
}
```

**Notes**
- Returns current active configuration
- If no config file exists, returns defaults

---

### Update Configuration

Update model configuration.

```
PUT /api/config
```

**Request Body**
```json
{
  "council_models": [
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-3-pro-preview"
  ],
  "chairman_model": "anthropic/claude-sonnet-4.5"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `council_models` | array | Yes | List of model identifiers (minimum 2) |
| `chairman_model` | string | Yes | Model identifier for the chairman |

**Response**
```json
{
  "council_models": [
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-3-pro-preview"
  ],
  "chairman_model": "anthropic/claude-sonnet-4.5"
}
```

**Error Responses**
| Status | Description |
|--------|-------------|
| 422 | Validation error (e.g., less than 2 models, empty strings) |

**Notes**
- Requires at least 2 council models
- All model identifiers must be non-empty strings
- Configuration is persisted to `data/council_config.json`
- Changes take effect immediately on next request

---

### Reset Configuration

Reset configuration to default values.

```
POST /api/config/reset
```

**Response**
```json
{
  "council_models": [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4"
  ],
  "chairman_model": "google/gemini-3-pro-preview"
}
```

---

### Get Available Models

Get list of suggested models for the configuration UI dropdown.

```
GET /api/config/models
```

**Response**
```json
{
  "models": [
    "openai/gpt-5.1",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-flash",
    "x-ai/grok-4",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1",
    "mistralai/mistral-large"
  ]
}
```

**Notes**
- Returns a curated list of popular OpenRouter models
- Used for autocomplete/suggestions in the UI
- Users can still enter any valid OpenRouter model identifier

---

**Event Types Reference**

| Type | Description | Data |
|------|-------------|------|
| `stage1_start` | Stage 1 beginning | None |
| `stage1_complete` | Stage 1 done | Array of responses (with usage/cost) |
| `stage2_start` | Stage 2 beginning | None |
| `stage2_complete` | Stage 2 done | Array of rankings + metadata (with usage/cost) |
| `stage3_start` | Stage 3 beginning | None |
| `stage3_complete` | Stage 3 done | Final answer object (with usage/cost) |
| `costs_complete` | Cost calculation done | Cost breakdown by stage and total |
| `title_complete` | Title generated | Title string |
| `complete` | All done | None |
| `error` | Error occurred | Error message |

**JavaScript Example**

```javascript
const response = await fetch(
  'http://localhost:8001/api/conversations/123/message/stream',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: 'Hello' }),
  }
);

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      console.log('Event:', event.type, event.data);
    }
  }
}
```

**Python Example**

```python
import httpx
import json

async def stream_message():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            'POST',
            'http://localhost:8001/api/conversations/123/message/stream',
            json={'content': 'Hello'},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    event = json.loads(line[6:])
                    print(f"Event: {event['type']}")
```

---

## Data Models

### Conversation Metadata

```json
{
  "id": "string (UUID)",
  "created_at": "string (ISO 8601 datetime)",
  "title": "string",
  "message_count": "integer",
  "tags": "array of strings"
}
```

### Full Conversation

```json
{
  "id": "string (UUID)",
  "created_at": "string (ISO 8601 datetime)",
  "title": "string",
  "tags": "array of strings",
  "messages": "array of Message"
}
```

### User Message

```json
{
  "role": "user",
  "content": "string"
}
```

### Assistant Message

```json
{
  "role": "assistant",
  "stage1": "array of Stage1Response",
  "stage2": "array of Stage2Response",
  "stage3": "Stage3Response"
}
```

### Stage1Response

```json
{
  "model": "string (model identifier)",
  "response": "string (model's answer)",
  "reasoning_details": "string | object | array (optional, for reasoning models)",
  "usage": {
    "prompt_tokens": "integer",
    "completion_tokens": "integer",
    "total_tokens": "integer"
  },
  "cost": {
    "input_cost": "number (USD)",
    "output_cost": "number (USD)",
    "total_cost": "number (USD)",
    "prompt_tokens": "integer",
    "completion_tokens": "integer",
    "total_tokens": "integer",
    "pricing": {"input": "number (per 1M)", "output": "number (per 1M)"}
  }
}
```

**Notes**
- `reasoning_details` is only present for reasoning models (o1, o3, DeepSeek-R1, etc.)
- `usage` and `cost` are included for all responses
- Frontend handles all formats and displays in a collapsible section

### Stage2Response

```json
{
  "model": "string (model identifier)",
  "ranking": "string (full evaluation text)",
  "parsed_ranking": "array of strings (e.g., ['Response A', 'Response C', ...])"
}
```

### Stage3Response

```json
{
  "model": "string (chairman model)",
  "response": "string (synthesized answer)"
}
```

### Metadata

```json
{
  "label_to_model": {
    "Response A": "model identifier",
    "Response B": "model identifier"
  },
  "aggregate_rankings": [
    {
      "model": "string",
      "average_rank": "number",
      "rankings_count": "integer"
    }
  ],
  "costs": {
    "stage1": {
      "total_cost": "number (USD)",
      "total_input_cost": "number (USD)",
      "total_output_cost": "number (USD)",
      "total_tokens": "integer",
      "total_prompt_tokens": "integer",
      "total_completion_tokens": "integer",
      "model_count": "integer"
    },
    "stage2": "same structure as stage1",
    "stage3": "same structure as stage1",
    "total": "same structure (aggregated across all stages)"
  }
}
```

> ⚠️ **Important**: Metadata is **ephemeral** - it is returned in API responses but **NOT persisted** to storage. When you retrieve a conversation via `GET /api/conversations/{id}`, the metadata will not be present. The frontend stores metadata in UI state during the session but it is lost on page refresh. This is by design: the `label_to_model` mapping, `aggregate_rankings`, and `costs` are computed fresh each time and are primarily for real-time display purposes.

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Resource not found |
| 422 | Validation error (invalid request body) |
| 500 | Internal server error |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8001/

# List conversations
curl http://localhost:8001/api/conversations

# Create conversation
curl -X POST http://localhost:8001/api/conversations \
  -H "Content-Type: application/json" \
  -d '{}'

# Get conversation
curl http://localhost:8001/api/conversations/{id}

# Send message (batch)
curl -X POST http://localhost:8001/api/conversations/{id}/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello"}'

# Send message with system prompt
curl -X POST http://localhost:8001/api/conversations/{id}/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello", "system_prompt": "You are a helpful assistant."}'

# Send message (streaming)
curl -X POST http://localhost:8001/api/conversations/{id}/message/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello"}'

# Send message (streaming) with system prompt
curl -X POST http://localhost:8001/api/conversations/{id}/message/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello", "system_prompt": "You are a helpful assistant."}'

# List conversations filtered by tag
curl "http://localhost:8001/api/conversations?tag=coding"

# Update tags for a conversation
curl -X PUT http://localhost:8001/api/conversations/{id}/tags \
  -H "Content-Type: application/json" \
  -d '{"tags": ["coding", "learning"]}'

# Get all unique tags
curl http://localhost:8001/api/tags

# Get current model configuration
curl http://localhost:8001/api/config

# Update model configuration
curl -X PUT http://localhost:8001/api/config \
  -H "Content-Type: application/json" \
  -d '{"council_models": ["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"], "chairman_model": "openai/gpt-5.1"}'

# Reset configuration to defaults
curl -X POST http://localhost:8001/api/config/reset

# Get available models for suggestions
curl http://localhost:8001/api/config/models
```

### Using Python requests

```python
import requests

# Create conversation
resp = requests.post('http://localhost:8001/api/conversations', json={})
conv = resp.json()
conv_id = conv['id']

# Send message
resp = requests.post(
    f'http://localhost:8001/api/conversations/{conv_id}/message',
    json={'content': 'What is Python?'}
)
result = resp.json()
print(result['stage3']['response'])

# Send message with system prompt
resp = requests.post(
    f'http://localhost:8001/api/conversations/{conv_id}/message',
    json={
        'content': 'Explain recursion',
        'system_prompt': 'You are a patient coding tutor. Use simple analogies.'
    }
)
result = resp.json()
print(result['stage3']['response'])
```

### Interactive API Docs

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## Rate Limiting

The API does not implement rate limiting internally. However, the OpenRouter API has its own rate limits based on your account tier.

If you encounter rate limit errors, they will appear in:
- Stage responses as `null` (model failed)
- SSE error events with details

---

## CORS Configuration

The backend allows requests from:
- `http://localhost:5173` (Vite dev server)
- `http://localhost:3000` (alternative dev server)

To add additional origins, modify `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://your-domain.com",  # Add new origin
    ],
    # ...
)
```
