# Extending the Codebase

This guide provides step-by-step instructions for common modifications and new features you might want to add.

## Table of Contents

1. [Adding a New Model to the Council](#adding-a-new-model-to-the-council)
2. [Changing the Chairman Model](#changing-the-chairman-model)
3. [Modifying the Stage 2 Ranking Prompt](#modifying-the-stage-2-ranking-prompt)
4. [Adding a New Stage](#adding-a-new-stage)
5. [Adding Conversation Export](#adding-conversation-export)
6. [Adding Model Configuration UI](#adding-model-configuration-ui)
7. [Adding Response Streaming](#adding-response-streaming)
8. [Adding User Authentication](#adding-user-authentication)
9. [Deploying to Production](#deploying-to-production)
10. [Common Patterns](#common-patterns)

---

## Adding a New Model to the Council

**Difficulty**: Easy
**Files to modify**: `backend/config.py`

### Steps

1. **Find the model identifier** on [openrouter.ai/models](https://openrouter.ai/models)

2. **Edit `backend/config.py`**:

```python
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
    "meta-llama/llama-3.1-405b-instruct",  # Add new model
]
```

3. **Restart the backend** (Ctrl+C and run again)

### Notes

- More models = more API costs and longer response times
- Each model adds 2 API calls per query (Stage 1 + Stage 2)
- The anonymization labels (A, B, C...) automatically expand

---

## Changing the Chairman Model

**Difficulty**: Easy
**Files to modify**: `backend/config.py`

### Steps

1. **Edit `backend/config.py`**:

```python
# Change from Gemini to Claude
CHAIRMAN_MODEL = "anthropic/claude-sonnet-4.5"
```

2. **Restart the backend**

### Considerations

- The chairman should be a capable model (it synthesizes all information)
- It can be the same as a council member or different
- Using a stronger model as chairman often produces better final answers

---

## Changing the Title Generation Model

**Difficulty**: Easy
**Files to modify**: `backend/council.py`

The title generation model is **hardcoded** (not in config.py) for simplicity.

### Steps

1. **Edit `backend/council.py`**, find line ~278 in `generate_conversation_title()`:

```python
# Current (hardcoded):
response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

# Change to any model you prefer:
response = await query_model("openai/gpt-4o-mini", messages, timeout=30.0)
```

2. **Restart the backend**

### Considerations

- Use a **fast, cheap model** - title generation happens on every first message
- The model only generates 3-5 words, so capability isn't critical
- If the model fails, title defaults to "New Conversation"
- Good choices: `google/gemini-2.5-flash`, `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`

### Making it Configurable

To add title model to config.py:

1. Add to `backend/config.py`:
```python
TITLE_MODEL = "google/gemini-2.5-flash"
```

2. Update `backend/council.py`:
```python
from .config import TITLE_MODEL

async def generate_conversation_title(user_query: str) -> str:
    # ...
    response = await query_model(TITLE_MODEL, messages, timeout=30.0)
```

---

## Modifying the Stage 2 Ranking Prompt

**Difficulty**: Medium
**Files to modify**: `backend/council.py`

The ranking prompt determines how models evaluate each other.

### Steps

1. **Edit `backend/council.py`**, find `stage2_collect_rankings()`:

```python
ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response on these criteria:
   - Accuracy: Is the information correct?
   - Completeness: Does it cover all important aspects?
   - Clarity: Is it easy to understand?
   - Practicality: Can the advice be applied?

2. Then, at the very end, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label

Example:
FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""
```

### Tips

- Keep the "FINAL RANKING:" format requirement - it's needed for parsing
- Be specific about evaluation criteria
- Test with several queries to ensure consistent parsing

---

## Adding a New Stage

**Difficulty**: Hard
**Files to modify**: Multiple

Let's add a "Stage 4: Fact Check" that verifies the final answer.

### Step 1: Backend - Add the new stage function

**Edit `backend/council.py`**:

```python
async def stage4_fact_check(
    user_query: str,
    stage3_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Stage 4: Fact-check the final answer.
    """
    fact_check_prompt = f"""You are a fact-checker. Review this answer for accuracy:

Original Question: {user_query}

Answer to Verify:
{stage3_result['response']}

Your task:
1. Identify any factual claims in the answer
2. Assess their accuracy
3. Note any potential errors or misleading statements
4. Provide a confidence score (1-10)

Format your response as:
CLAIMS:
- [claim 1]: [assessment]
- [claim 2]: [assessment]

CONFIDENCE SCORE: [1-10]

SUMMARY: [brief overall assessment]"""

    messages = [{"role": "user", "content": fact_check_prompt}]
    response = await query_model("anthropic/claude-sonnet-4.5", messages)

    if response is None:
        return {"model": "anthropic/claude-sonnet-4.5", "response": "Fact check failed."}

    return {
        "model": "anthropic/claude-sonnet-4.5",
        "response": response.get('content', '')
    }
```

### Step 2: Backend - Update run_full_council

**Edit `backend/council.py`**:

```python
async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict, Dict]:
    """Now returns 5 items instead of 4"""
    # ... existing stage 1, 2, 3 code ...

    # Stage 4: Fact check
    stage4_result = await stage4_fact_check(user_query, stage3_result)

    return stage1_results, stage2_results, stage3_result, stage4_result, metadata
```

### Step 3: Backend - Update API endpoint

**Edit `backend/main.py`**:

```python
@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    # ... existing code ...

    async def event_generator():
        # ... existing stage 1, 2, 3 code ...

        # Add Stage 4
        yield f"data: {json.dumps({'type': 'stage4_start'})}\n\n"
        stage4_result = await stage4_fact_check(request.content, stage3_result)
        yield f"data: {json.dumps({'type': 'stage4_complete', 'data': stage4_result})}\n\n"

        # Update storage call
        storage.add_assistant_message(
            conversation_id,
            stage1_results,
            stage2_results,
            stage3_result,
            stage4_result  # Add new parameter
        )
```

### Step 4: Backend - Update storage

**Edit `backend/storage.py`**:

```python
def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any],
    stage4: Dict[str, Any] = None  # Optional for backwards compatibility
):
    conversation["messages"].append({
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4
    })
```

### Step 5: Frontend - Create Stage4 component

**Create `frontend/src/components/Stage4.jsx`**:

```jsx
import ReactMarkdown from 'react-markdown';
import './Stage4.css';

export default function Stage4({ response }) {
  if (!response) return null;

  return (
    <div className="stage stage4">
      <h3 className="stage-title">Stage 4: Fact Check</h3>
      <div className="fact-check-model">Verified by: {response.model}</div>
      <div className="fact-check-content markdown-content">
        <ReactMarkdown>{response.response}</ReactMarkdown>
      </div>
    </div>
  );
}
```

### Step 6: Frontend - Update ChatInterface

**Edit `frontend/src/components/ChatInterface.jsx`**:

```jsx
import Stage4 from './Stage4';

// In the render, add after Stage3:
{message.loading?.stage4 && (
  <div className="loading-indicator">
    <span className="spinner"></span>
    Stage 4: Fact checking...
  </div>
)}
{message.stage4 && <Stage4 response={message.stage4} />}
```

### Step 7: Frontend - Handle SSE events

**Edit `frontend/src/App.jsx`**, add cases in the switch:

```jsx
case 'stage4_start':
  setCurrentConversation((prev) => {
    const messages = [...prev.messages];
    const lastMsg = messages[messages.length - 1];
    lastMsg.loading.stage4 = true;
    return { ...prev, messages };
  });
  break;

case 'stage4_complete':
  setCurrentConversation((prev) => {
    const messages = [...prev.messages];
    const lastMsg = messages[messages.length - 1];
    lastMsg.stage4 = event.data;
    lastMsg.loading.stage4 = false;
    return { ...prev, messages };
  });
  break;
```

---

## Adding Conversation Export

**Difficulty**: Medium
**Files to modify**: `backend/main.py`, frontend components

### Step 1: Backend - Add export endpoint

**Edit `backend/main.py`**:

```python
from fastapi.responses import Response

@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str, format: str = "markdown"):
    """Export a conversation as markdown or JSON."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if format == "json":
        return conversation

    # Convert to markdown
    md = f"# {conversation['title']}\n\n"
    md += f"*Created: {conversation['created_at']}*\n\n---\n\n"

    for msg in conversation['messages']:
        if msg['role'] == 'user':
            md += f"## User\n\n{msg['content']}\n\n"
        else:
            md += "## Council Response\n\n"
            if msg.get('stage3'):
                md += f"### Final Answer\n\n{msg['stage3']['response']}\n\n"

    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={conversation_id}.md"}
    )
```

### Step 2: Frontend - Add export button

**Edit `frontend/src/components/ChatInterface.jsx`**:

```jsx
const handleExport = () => {
  if (!conversation) return;
  window.open(
    `http://localhost:8001/api/conversations/${conversation.id}/export?format=markdown`,
    '_blank'
  );
};

// In the render:
<button className="export-btn" onClick={handleExport}>
  Export
</button>
```

---

## Adding Model Configuration UI

**Difficulty**: Medium-Hard
**Files to modify**: Multiple

Allow users to select models from the frontend.

### Step 1: Backend - Add configuration endpoint

**Edit `backend/main.py`**:

```python
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL

@app.get("/api/config")
async def get_config():
    """Get current model configuration."""
    return {
        "council_models": COUNCIL_MODELS,
        "chairman_model": CHAIRMAN_MODEL,
        "available_models": [
            "openai/gpt-5.1",
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-3-opus",
            "google/gemini-3-pro-preview",
            "google/gemini-2.5-flash",
            "x-ai/grok-4",
            "meta-llama/llama-3.1-405b-instruct",
        ]
    }
```

### Step 2: Frontend - Create settings component

Create a settings modal that lets users modify the configuration. Store selections in localStorage and pass to API calls.

---

## Adding Response Streaming

**Difficulty**: Hard
**Files to modify**: Multiple

Stream individual model responses word-by-word instead of waiting for completion.

### Concept

OpenRouter supports streaming responses. You would:
1. Use `stream: true` in the API request
2. Process Server-Sent Events from OpenRouter
3. Forward them to the frontend as they arrive

### Key Changes

**Backend** (`openrouter.py`):
```python
async def query_model_streaming(model, messages, on_token):
    payload = {"model": model, "messages": messages, "stream": True}
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    token = data["choices"][0]["delta"].get("content", "")
                    await on_token(token)
```

This is more complex as you need to handle multiple parallel streams and aggregate them.

---

## Adding User Authentication

**Difficulty**: Hard
**Files to modify**: Multiple

For a simple approach, use session-based auth with cookies.

### Backend Changes

1. **Add dependencies**:
```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

2. **Create auth module** (`backend/auth.py`):
```python
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext

SECRET_KEY = "your-secret-key"  # Move to .env
pwd_context = CryptContext(schemes=["bcrypt"])

def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY)

def verify_token(token: str) -> str:
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload["sub"]
```

3. **Add auth middleware and endpoints**

### Frontend Changes

1. Add login/register forms
2. Store JWT in localStorage
3. Include in API requests

---

## Deploying to Production

**Difficulty**: Medium

### Option 1: Docker

**Create `Dockerfile`**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv sync

COPY backend/ backend/
CMD ["uv", "run", "python", "-m", "backend.main"]
```

**Create `docker-compose.yml`**:
```yaml
version: '3'
services:
  backend:
    build: .
    ports:
      - "8001:8001"
    env_file:
      - .env

  frontend:
    build: ./frontend
    ports:
      - "80:80"
```

### Option 2: Cloud Platform

Deploy to Railway, Render, or similar:

1. Push code to GitHub
2. Connect repository to platform
3. Set environment variables
4. Deploy

---

## Common Patterns

### Adding a New API Endpoint

1. Define the endpoint in `backend/main.py`
2. Add any new Pydantic models for request/response
3. Implement the logic (possibly in a separate module)
4. Add to `frontend/src/api.js`
5. Use in frontend components

### Adding a New Component

1. Create `ComponentName.jsx` and `ComponentName.css` in `frontend/src/components/`
2. Import and use in parent component
3. Pass data via props
4. Handle events via callback props

### Adding State

1. If local to one component: use `useState` in that component
2. If shared across siblings: lift state to common parent (currently App.jsx)
3. If complex/global: consider React Context or a state library

### Making API Calls

```javascript
// In api.js
async newFunction(param) {
  const response = await fetch(`${API_BASE}/api/endpoint`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ param }),
  });
  if (!response.ok) throw new Error('Failed');
  return response.json();
}

// In component
const handleClick = async () => {
  try {
    const result = await api.newFunction(value);
    // Update state with result
  } catch (error) {
    console.error(error);
  }
};
```

---

## Tips for Success

1. **Make small changes** - Test after each modification
2. **Check the console** - Browser DevTools and terminal show errors
3. **Use the API docs** - Visit http://localhost:8001/docs
4. **Keep backups** - Use git to track changes
5. **Read error messages** - They usually point to the exact problem

---

## Need More Help?

- Review [Architecture](./ARCHITECTURE.md) to understand how pieces fit together
- Check [Troubleshooting](./TROUBLESHOOTING.md) for common issues
- Read the code comments in the source files
