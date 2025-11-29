# Feature Ideas for LLM Council

This document contains 21 feature suggestions for extending LLM Council. Each feature is designed to be meaningful, useful, and educational - helping you learn important concepts in LLM application development.

The features are organized into two categories:
1. **General Features** (11) - UI improvements, data management, user experience
2. **Orchestration Layer Features** (10) - Core coordination logic between models

---

## Table of Contents

### Part 1: General Features
1. [Model Performance Dashboard](#1-model-performance-dashboard)
2. [Conversation Export](#2-conversation-export-markdownpdf)
3. [Custom System Prompts](#3-custom-system-prompts)
4. [Streaming Responses](#4-streaming-responses)
5. [Model Configuration UI](#5-model-configuration-ui)
6. [Question Categories & Tagging](#6-question-categories--tagging)
7. [Reasoning Model Support](#7-reasoning-model-support)
8. [Cost Tracking](#8-cost-tracking)
9. [Debate Mode](#9-debate-mode)
10. [Confidence Voting](#10-confidence-voting)
11. [Live Process Monitor](#11-live-process-monitor)

### Part 2: Orchestration Layer Features
12. [Dynamic Model Routing](#12-dynamic-model-routing)
13. [Weighted Consensus Voting](#13-weighted-consensus-voting)
14. [Early Consensus Exit](#14-early-consensus-exit)
15. [Iterative Refinement Loop](#15-iterative-refinement-loop)
16. [Parallel Sub-Question Decomposition](#16-parallel-sub-question-decomposition)
17. [Adversarial Validation Stage](#17-adversarial-validation-stage)
18. [Confidence-Gated Escalation](#18-confidence-gated-escalation)
19. [Response Caching Layer](#19-response-caching-layer)
20. [Chain-of-Thought Orchestration](#20-chain-of-thought-orchestration)
21. [Multi-Chairman Synthesis](#21-multi-chairman-synthesis)

### Reference
- [Complexity Summary](#complexity-summary)
- [Recommended Learning Paths](#recommended-learning-paths)
- [Feature Dependencies](#feature-dependencies)

---

# Part 1: General Features

These features enhance the user experience, add practical functionality, and teach fundamental web development patterns in the context of LLM applications.

---

## 1. Model Performance Dashboard

### Overview
Track and visualize how often each model gets ranked #1, #2, etc. over time. Display statistics showing which models consistently perform best.

### Why It's Valuable
- **For users**: See which models are worth keeping in the council
- **For learning**: Understand data aggregation and visualization patterns
- **For optimization**: Make data-driven decisions about model selection

### What You'll Learn
- Data aggregation patterns
- Charting libraries (Chart.js, Recharts, or D3.js)
- localStorage or database persistence
- Statistical calculations (averages, distributions)

### Implementation Approach

#### Backend Changes

**New file: `backend/analytics.py`**
```python
from typing import Dict, List
from datetime import datetime
import json
import os

ANALYTICS_FILE = "data/analytics.json"

def record_ranking(question_id: str, rankings: Dict[str, List[str]], label_to_model: Dict[str, str]):
    """
    Record ranking results for later analysis.

    Args:
        question_id: Unique identifier for this question
        rankings: Dict mapping model -> their ranking list
        label_to_model: Dict mapping "Response A" -> model identifier
    """
    # Load existing data
    data = load_analytics()

    # Create ranking record
    record = {
        "timestamp": datetime.now().isoformat(),
        "question_id": question_id,
        "rankings": rankings,
        "label_to_model": label_to_model
    }

    data["ranking_history"].append(record)
    save_analytics(data)

def get_model_statistics() -> Dict[str, Dict]:
    """
    Calculate performance statistics for each model.

    Returns:
        Dict mapping model -> {
            "times_ranked_1": int,
            "times_ranked_2": int,
            "average_rank": float,
            "total_appearances": int
        }
    """
    data = load_analytics()
    stats = {}

    for record in data["ranking_history"]:
        label_to_model = record["label_to_model"]

        for ranker_model, ranking_list in record["rankings"].items():
            for position, label in enumerate(ranking_list, 1):
                model = label_to_model.get(label)
                if model:
                    if model not in stats:
                        stats[model] = {
                            "times_ranked_1": 0,
                            "times_ranked_2": 0,
                            "times_ranked_3": 0,
                            "times_ranked_4": 0,
                            "total_rank_sum": 0,
                            "total_appearances": 0
                        }

                    stats[model][f"times_ranked_{position}"] = stats[model].get(f"times_ranked_{position}", 0) + 1
                    stats[model]["total_rank_sum"] += position
                    stats[model]["total_appearances"] += 1

    # Calculate averages
    for model in stats:
        if stats[model]["total_appearances"] > 0:
            stats[model]["average_rank"] = (
                stats[model]["total_rank_sum"] / stats[model]["total_appearances"]
            )

    return stats
```

**Add endpoint in `backend/main.py`**
```python
from .analytics import get_model_statistics, record_ranking

@app.get("/api/analytics/model-performance")
async def get_model_performance():
    """Get performance statistics for all models."""
    return get_model_statistics()
```

#### Frontend Changes

**New file: `frontend/src/components/PerformanceDashboard.jsx`**
```jsx
import { useState, useEffect } from 'react';
import { getModelPerformance } from '../api';

// Option 1: Simple table view
// Option 2: Bar chart using recharts
// Option 3: Pie chart showing rank distribution

export function PerformanceDashboard() {
    const [stats, setStats] = useState({});

    useEffect(() => {
        getModelPerformance().then(setStats);
    }, []);

    return (
        <div className="dashboard">
            <h2>Model Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>#1 Ranks</th>
                        <th>#2 Ranks</th>
                        <th>Avg Rank</th>
                        <th>Total</th>
                    </tr>
                </thead>
                <tbody>
                    {Object.entries(stats).map(([model, data]) => (
                        <tr key={model}>
                            <td>{model}</td>
                            <td>{data.times_ranked_1}</td>
                            <td>{data.times_ranked_2}</td>
                            <td>{data.average_rank?.toFixed(2)}</td>
                            <td>{data.total_appearances}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/analytics.py` | Create | Analytics data collection and calculation |
| `backend/main.py` | Modify | Add `/api/analytics/model-performance` endpoint |
| `backend/council.py` | Modify | Call `record_ranking()` after Stage 2 |
| `frontend/src/api.js` | Modify | Add `getModelPerformance()` function |
| `frontend/src/components/PerformanceDashboard.jsx` | Create | Dashboard UI component |
| `frontend/src/App.jsx` | Modify | Add route/tab for dashboard |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 6
**New Concepts**: Data persistence, aggregation, visualization

---

## 2. Conversation Export (Markdown/PDF) ✅ IMPLEMENTED

> **Status**: This feature has been implemented! A header toolbar with "Export MD" and "Export JSON" buttons appears when a conversation has messages. Uses client-side Blob API for file generation.

### Overview
Export a conversation with all stages to a shareable document format (Markdown or PDF).

### Why It's Valuable
- **For users**: Share council deliberations with colleagues
- **For learning**: Understand document generation and file downloads
- **For documentation**: Create reports from AI consultations

### What You'll Learn
- File generation in the browser
- Blob creation and download triggers
- Document formatting and templating
- (Optional) PDF generation libraries

### Implementation Approach

#### Backend Approach (Server-side generation)

**Add to `backend/main.py`**
```python
from fastapi.responses import PlainTextResponse

@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str, format: str = "markdown"):
    """Export conversation to markdown or other formats."""
    conversation = storage.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if format == "markdown":
        return PlainTextResponse(
            content=generate_markdown(conversation),
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={conversation_id}.md"
            }
        )

def generate_markdown(conversation: dict) -> str:
    """Convert conversation to markdown format."""
    lines = [
        f"# {conversation.get('title', 'Conversation')}",
        f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    for msg in conversation.get("messages", []):
        if msg["role"] == "user":
            lines.append(f"## User Question")
            lines.append(msg["content"])
            lines.append("")
        else:
            # Stage 1
            lines.append("## Stage 1: Individual Responses")
            for resp in msg.get("stage1", []):
                lines.append(f"### {resp['model']}")
                lines.append(resp["response"])
                lines.append("")

            # Stage 2
            lines.append("## Stage 2: Peer Rankings")
            for ranking in msg.get("stage2", []):
                lines.append(f"### {ranking['model']}'s Evaluation")
                lines.append(ranking["ranking"])
                lines.append("")

            # Stage 3
            lines.append("## Stage 3: Final Synthesis")
            stage3 = msg.get("stage3", {})
            lines.append(f"*Chairman: {stage3.get('model', 'Unknown')}*")
            lines.append("")
            lines.append(stage3.get("response", ""))
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)
```

#### Frontend Approach (Client-side generation)

**Add to `frontend/src/utils/export.js`**
```javascript
export function exportToMarkdown(conversation) {
    const markdown = generateMarkdown(conversation);
    downloadFile(markdown, `${conversation.title || 'conversation'}.md`, 'text/markdown');
}

function generateMarkdown(conversation) {
    let md = `# ${conversation.title || 'Conversation'}\n\n`;
    md += `*Exported: ${new Date().toLocaleString()}*\n\n`;

    for (const msg of conversation.messages) {
        if (msg.role === 'user') {
            md += `## User Question\n\n${msg.content}\n\n`;
        } else {
            // Stage 1
            md += `## Stage 1: Individual Responses\n\n`;
            for (const resp of msg.stage1 || []) {
                md += `### ${resp.model}\n\n${resp.response}\n\n`;
            }

            // Stage 2
            md += `## Stage 2: Peer Rankings\n\n`;
            for (const ranking of msg.stage2 || []) {
                md += `### ${ranking.model}'s Evaluation\n\n${ranking.ranking}\n\n`;
            }

            // Stage 3
            md += `## Stage 3: Final Synthesis\n\n`;
            md += `*Chairman: ${msg.stage3?.model}*\n\n`;
            md += `${msg.stage3?.response}\n\n`;
            md += `---\n\n`;
        }
    }

    return md;
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
```

**Add export button to UI**
```jsx
// In ChatInterface.jsx or a toolbar component
import { exportToMarkdown } from '../utils/export';

<button onClick={() => exportToMarkdown(currentConversation)}>
    Export to Markdown
</button>
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `frontend/src/utils/export.js` | Create | Export utility functions |
| `frontend/src/components/ChatInterface.jsx` | Modify | Add export button |
| (Optional) `backend/main.py` | Modify | Server-side export endpoint |

### Estimated Complexity
**Difficulty**: Easy
**Files Changed**: 2-3
**New Concepts**: Blob API, file downloads, document templating

---

## 3. Custom System Prompts ✅ IMPLEMENTED

> **Status**: This feature has been implemented! A collapsible settings panel in the UI allows users to enter a system prompt that is prepended to all council model queries. The prompt is persisted to localStorage.

### Overview
Let users define a system prompt (persona/instructions) that's prepended to all model queries. For example: "You are an expert software architect. Always consider scalability."

### Why It's Valuable
- **For users**: Tailor the council for specific domains (legal, medical, coding)
- **For learning**: Understand how system prompts shape LLM behavior
- **For experimentation**: Test different prompting strategies

### What You'll Learn
- Prompt engineering fundamentals
- How system prompts affect model outputs
- State management for user preferences
- Message structure in LLM APIs

### Implementation Approach

#### Backend Changes

**Modify `backend/council.py`**
```python
async def stage1_collect_responses(
    question: str,
    conversation_history: List[Dict] = None,
    system_prompt: str = None  # NEW PARAMETER
) -> List[Dict[str, Any]]:
    """Collect responses from all council models."""

    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)

    # Add current question
    messages.append({
        "role": "user",
        "content": question
    })

    # Query all models
    responses = await query_models_parallel(COUNCIL_MODELS, messages)
    # ... rest of function
```

**Modify API endpoint in `backend/main.py`**
```python
class MessageRequest(BaseModel):
    content: str
    system_prompt: Optional[str] = None  # NEW FIELD

@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: MessageRequest):
    # Pass system_prompt to council functions
    stage1_responses = await stage1_collect_responses(
        request.content,
        conversation_history,
        system_prompt=request.system_prompt  # NEW
    )
```

#### Frontend Changes

**Add system prompt input to UI**
```jsx
// In App.jsx or a settings component
const [systemPrompt, setSystemPrompt] = useState(
    localStorage.getItem('systemPrompt') || ''
);

// Settings panel
<div className="settings-panel">
    <label>System Prompt (Optional)</label>
    <textarea
        value={systemPrompt}
        onChange={(e) => {
            setSystemPrompt(e.target.value);
            localStorage.setItem('systemPrompt', e.target.value);
        }}
        placeholder="You are an expert software architect..."
        rows={4}
    />
</div>

// Pass to API call
const handleSendMessage = async (content) => {
    await sendMessageStream(conversationId, content, systemPrompt);
};
```

**Modify `frontend/src/api.js`**
```javascript
export async function sendMessageStream(conversationId, content, systemPrompt = null) {
    const response = await fetch(
        `${API_BASE}/api/conversations/${conversationId}/message/stream`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                content,
                system_prompt: systemPrompt  // NEW
            }),
        }
    );
    // ... rest of function
}
```

### Example System Prompts

```
# For coding questions
You are an expert software engineer. Prioritize code quality, maintainability, and best practices. Always consider edge cases and error handling.

# For writing help
You are a professional editor. Focus on clarity, concision, and engaging prose. Suggest improvements while preserving the author's voice.

# For research questions
You are a research analyst. Cite sources when possible, acknowledge uncertainty, and present multiple perspectives on controversial topics.

# For learning
You are a patient teacher. Explain concepts step by step, use analogies, and check for understanding. Adapt to the learner's level.
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/council.py` | Modify | Add system_prompt parameter to stage functions |
| `backend/main.py` | Modify | Accept system_prompt in request body |
| `frontend/src/api.js` | Modify | Pass system_prompt to API |
| `frontend/src/App.jsx` | Modify | Add system prompt input UI |
| `frontend/src/components/Settings.jsx` | Create | (Optional) Dedicated settings panel |

### Estimated Complexity
**Difficulty**: Easy
**Files Changed**: 4
**New Concepts**: Prompt engineering, system messages, localStorage

---

## 4. Streaming Responses

### Overview
Show model responses as they generate (word by word) instead of waiting for completion. Currently, users wait 20-60 seconds seeing nothing; with streaming, they see words appear in real-time.

### Why It's Valuable
- **For users**: Much better UX - see progress immediately
- **For learning**: Deep understanding of Server-Sent Events and streaming
- **For perception**: Responses feel faster even if total time is the same

### What You'll Learn
- Token streaming from LLM APIs
- Server-Sent Events (SSE) for real-time data
- Incremental UI updates
- Async generators in Python
- Stream processing in JavaScript

### Implementation Approach

This is a significant refactor. The current flow is:
1. Query model → Wait for complete response → Send to frontend

The streaming flow is:
1. Query model with streaming → Forward each token to frontend → Accumulate final response

#### Backend Changes

**Modify `backend/openrouter.py`**
```python
async def query_model_streaming(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
):
    """
    Query a model with streaming, yielding tokens as they arrive.

    Yields:
        dict with 'token' (str) or 'done' (bool) or 'error' (str)
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,  # Enable streaming
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield {"done": True}
                            break

                        try:
                            chunk = json.loads(data)
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token:
                                yield {"token": token}
                        except (json.JSONDecodeError, KeyError):
                            continue

    except Exception as e:
        yield {"error": str(e)}
```

**Modify streaming endpoint in `backend/main.py`**
```python
@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: MessageRequest):
    async def event_generator():
        # Stage 1 with per-model streaming
        yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"

        stage1_responses = []
        for model in COUNCIL_MODELS:
            yield f"data: {json.dumps({'type': 'model_start', 'model': model})}\n\n"

            full_response = ""
            async for chunk in query_model_streaming(model, messages):
                if "token" in chunk:
                    full_response += chunk["token"]
                    yield f"data: {json.dumps({'type': 'token', 'model': model, 'token': chunk['token']})}\n\n"
                elif "done" in chunk:
                    stage1_responses.append({"model": model, "response": full_response})
                    yield f"data: {json.dumps({'type': 'model_complete', 'model': model})}\n\n"

        yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_responses})}\n\n"

        # Continue with Stage 2, 3...

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

#### Frontend Changes

**Modify `frontend/src/api.js`**
```javascript
export async function* streamMessage(conversationId, content) {
    const response = await fetch(
        `${API_BASE}/api/conversations/${conversationId}/message/stream`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
        }
    );

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const event = JSON.parse(line.slice(6));
                yield event;
            }
        }
    }
}
```

**Handle streaming in component**
```jsx
const handleSendMessage = async (content) => {
    const streamingResponses = {}; // Track partial responses per model

    for await (const event of streamMessage(conversationId, content)) {
        switch (event.type) {
            case 'token':
                // Append token to model's response
                streamingResponses[event.model] =
                    (streamingResponses[event.model] || '') + event.token;
                // Update UI to show partial response
                setPartialResponses({...streamingResponses});
                break;

            case 'model_complete':
                // Model finished streaming
                break;

            case 'stage1_complete':
                // All Stage 1 done, clear partial, set final
                setStage1(event.data);
                setPartialResponses({});
                break;

            // ... handle other events
        }
    }
};
```

### Architecture Diagram

```
Current (Batch):
┌────────┐    wait 30s    ┌────────┐
│ Client │ ──────────────▶│ Server │──▶ OpenRouter ──▶ Complete response
└────────┘                └────────┘

Streaming:
┌────────┐    token 1     ┌────────┐
│ Client │ ◀───────────── │ Server │◀── OpenRouter ◀── "The"
│        │    token 2     │        │◀── OpenRouter ◀── " answer"
│        │ ◀───────────── │        │◀── OpenRouter ◀── " is"
│        │    token 3     │        │◀── OpenRouter ◀── "..."
│        │ ◀───────────── │        │
└────────┘                └────────┘
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/openrouter.py` | Modify | Add `query_model_streaming()` function |
| `backend/main.py` | Modify | Refactor stream endpoint for token-level events |
| `frontend/src/api.js` | Modify | Use async generator for streaming |
| `frontend/src/App.jsx` | Modify | Handle incremental token updates |
| `frontend/src/components/Stage1.jsx` | Modify | Display partial responses during streaming |

### Estimated Complexity
**Difficulty**: Hard
**Files Changed**: 5
**New Concepts**: Token streaming, async generators, incremental rendering

### Notes
- Not all models support streaming equally well
- Stage 2 and 3 can also be streamed using the same pattern
- Consider showing a typing indicator per model

---

## 5. Model Configuration UI

### Overview
Add/remove council models and change the chairman through the web interface instead of editing `config.py`.

### Why It's Valuable
- **For users**: Experiment with different model combinations easily
- **For learning**: Full-stack form handling, state persistence
- **For flexibility**: No code changes needed to try new models

### What You'll Learn
- RESTful API design for configuration
- Form handling in React
- State management patterns
- Configuration persistence

### Implementation Approach

#### Backend Changes

**Create `backend/config_api.py`**
```python
from pydantic import BaseModel
from typing import List
import json
import os

CONFIG_FILE = "data/config.json"

class CouncilConfig(BaseModel):
    council_models: List[str]
    chairman_model: str

def load_config() -> CouncilConfig:
    """Load config from file or return defaults."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            return CouncilConfig(**data)

    # Default config
    return CouncilConfig(
        council_models=[
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
            "x-ai/grok-beta",
        ],
        chairman_model="google/gemini-pro-1.5"
    )

def save_config(config: CouncilConfig):
    """Save config to file."""
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config.dict(), f, indent=2)

def get_council_models() -> List[str]:
    """Get current council models."""
    return load_config().council_models

def get_chairman_model() -> str:
    """Get current chairman model."""
    return load_config().chairman_model
```

**Add endpoints to `backend/main.py`**
```python
from .config_api import CouncilConfig, load_config, save_config

@app.get("/api/config")
async def get_config():
    """Get current council configuration."""
    config = load_config()
    return config.dict()

@app.put("/api/config")
async def update_config(config: CouncilConfig):
    """Update council configuration."""
    # Validate models exist (optional: query OpenRouter)
    if len(config.council_models) < 2:
        raise HTTPException(400, "Need at least 2 council models")

    if config.chairman_model not in config.council_models:
        # Chairman doesn't have to be in council, but warn
        pass

    save_config(config)
    return {"status": "ok", "config": config.dict()}

@app.get("/api/models/available")
async def get_available_models():
    """Get list of available models from OpenRouter."""
    # Could query OpenRouter's model list API
    # For now, return a curated list
    return {
        "models": [
            {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI"},
            {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "Anthropic"},
            {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku", "provider": "Anthropic"},
            {"id": "google/gemini-pro-1.5", "name": "Gemini 1.5 Pro", "provider": "Google"},
            {"id": "google/gemini-flash-1.5", "name": "Gemini 1.5 Flash", "provider": "Google"},
            {"id": "x-ai/grok-beta", "name": "Grok", "provider": "xAI"},
            {"id": "meta-llama/llama-3.1-405b-instruct", "name": "Llama 3.1 405B", "provider": "Meta"},
        ]
    }
```

**Update `backend/council.py` to use dynamic config**
```python
from .config_api import get_council_models, get_chairman_model

async def stage1_collect_responses(question: str, ...):
    models = get_council_models()  # Dynamic instead of imported constant
    responses = await query_models_parallel(models, messages)
    # ...

async def stage3_synthesize_final(...):
    chairman = get_chairman_model()  # Dynamic
    response = await query_model(chairman, messages)
    # ...
```

#### Frontend Changes

**Create `frontend/src/components/ConfigPanel.jsx`**
```jsx
import { useState, useEffect } from 'react';
import { getConfig, updateConfig, getAvailableModels } from '../api';

export function ConfigPanel() {
    const [config, setConfig] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        getConfig().then(setConfig);
        getAvailableModels().then(data => setAvailableModels(data.models));
    }, []);

    const toggleModel = (modelId) => {
        const models = config.council_models.includes(modelId)
            ? config.council_models.filter(m => m !== modelId)
            : [...config.council_models, modelId];
        setConfig({ ...config, council_models: models });
    };

    const handleSave = async () => {
        setSaving(true);
        await updateConfig(config);
        setSaving(false);
    };

    if (!config) return <div>Loading...</div>;

    return (
        <div className="config-panel">
            <h2>Council Configuration</h2>

            <h3>Council Members</h3>
            <p>Select 2-6 models to participate in the council:</p>
            {availableModels.map(model => (
                <label key={model.id}>
                    <input
                        type="checkbox"
                        checked={config.council_models.includes(model.id)}
                        onChange={() => toggleModel(model.id)}
                    />
                    {model.name} ({model.provider})
                </label>
            ))}

            <h3>Chairman Model</h3>
            <p>Select the model that synthesizes the final answer:</p>
            <select
                value={config.chairman_model}
                onChange={(e) => setConfig({ ...config, chairman_model: e.target.value })}
            >
                {availableModels.map(model => (
                    <option key={model.id} value={model.id}>
                        {model.name}
                    </option>
                ))}
            </select>

            <button onClick={handleSave} disabled={saving}>
                {saving ? 'Saving...' : 'Save Configuration'}
            </button>
        </div>
    );
}
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/config_api.py` | Create | Configuration management functions |
| `backend/main.py` | Modify | Add config API endpoints |
| `backend/council.py` | Modify | Use dynamic config instead of constants |
| `frontend/src/api.js` | Modify | Add config API functions |
| `frontend/src/components/ConfigPanel.jsx` | Create | Configuration UI |
| `frontend/src/App.jsx` | Modify | Add config panel route/tab |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 6
**New Concepts**: Configuration management, form handling, RESTful design

---

## 6. Question Categories & Tagging ✅ IMPLEMENTED

> **Status**: This feature has been implemented! Tag filter dropdown in sidebar, tags displayed on conversations, TagEditor component for adding/removing tags, and suggested tags for quick selection.

### Overview
Tag conversations by topic (coding, writing, analysis, etc.) and filter/search by tags.

### Why It's Valuable
- **For users**: Organize conversations, find past discussions easily
- **For learning**: Data modeling, search/filter patterns
- **For analysis**: Discover which council performs best per category

### What You'll Learn
- Data modeling with metadata
- Search and filter UI patterns
- Tag management (add, remove, suggest)
- Faceted search concepts

### Implementation Approach

#### Backend Changes

**Update conversation schema in `backend/storage.py`**
```python
def create_conversation() -> Dict[str, Any]:
    """Create a new conversation."""
    conversation = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "title": "New Conversation",
        "tags": [],  # NEW: List of tag strings
        "messages": []
    }
    save_conversation(conversation)
    return conversation

def update_conversation_tags(conversation_id: str, tags: List[str]) -> Dict:
    """Update tags for a conversation."""
    conversation = get_conversation(conversation_id)
    if conversation:
        conversation["tags"] = tags
        save_conversation(conversation)
    return conversation

def get_all_tags() -> List[str]:
    """Get all unique tags across conversations."""
    tags = set()
    for conv in list_conversations():
        full_conv = get_conversation(conv["id"])
        tags.update(full_conv.get("tags", []))
    return sorted(list(tags))

def filter_conversations_by_tag(tag: str) -> List[Dict]:
    """Get conversations with a specific tag."""
    result = []
    for conv in list_conversations():
        full_conv = get_conversation(conv["id"])
        if tag in full_conv.get("tags", []):
            result.append(conv)
    return result
```

**Add endpoints to `backend/main.py`**
```python
@app.put("/api/conversations/{conversation_id}/tags")
async def update_tags(conversation_id: str, tags: List[str]):
    """Update tags for a conversation."""
    conversation = storage.update_conversation_tags(conversation_id, tags)
    if not conversation:
        raise HTTPException(404, "Conversation not found")
    return {"status": "ok", "tags": tags}

@app.get("/api/tags")
async def get_all_tags():
    """Get all unique tags."""
    return {"tags": storage.get_all_tags()}

@app.get("/api/conversations")
async def list_conversations(tag: Optional[str] = None):
    """List conversations, optionally filtered by tag."""
    if tag:
        return storage.filter_conversations_by_tag(tag)
    return storage.list_conversations()
```

#### Frontend Changes

**Create `frontend/src/components/TagEditor.jsx`**
```jsx
import { useState } from 'react';

const SUGGESTED_TAGS = [
    'coding', 'writing', 'analysis', 'research',
    'creative', 'technical', 'business', 'learning'
];

export function TagEditor({ tags, onTagsChange }) {
    const [inputValue, setInputValue] = useState('');

    const addTag = (tag) => {
        if (tag && !tags.includes(tag)) {
            onTagsChange([...tags, tag]);
        }
        setInputValue('');
    };

    const removeTag = (tag) => {
        onTagsChange(tags.filter(t => t !== tag));
    };

    return (
        <div className="tag-editor">
            <div className="tags">
                {tags.map(tag => (
                    <span key={tag} className="tag">
                        {tag}
                        <button onClick={() => removeTag(tag)}>×</button>
                    </span>
                ))}
            </div>

            <input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addTag(inputValue)}
                placeholder="Add tag..."
            />

            <div className="suggested-tags">
                {SUGGESTED_TAGS.filter(t => !tags.includes(t)).map(tag => (
                    <button key={tag} onClick={() => addTag(tag)}>
                        + {tag}
                    </button>
                ))}
            </div>
        </div>
    );
}
```

**Add tag filter to sidebar**
```jsx
// In Sidebar.jsx or ConversationList.jsx
const [selectedTag, setSelectedTag] = useState(null);
const [allTags, setAllTags] = useState([]);

useEffect(() => {
    getAllTags().then(data => setAllTags(data.tags));
}, []);

// Filter UI
<div className="tag-filter">
    <select value={selectedTag || ''} onChange={(e) => setSelectedTag(e.target.value || null)}>
        <option value="">All conversations</option>
        {allTags.map(tag => (
            <option key={tag} value={tag}>{tag}</option>
        ))}
    </select>
</div>
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/storage.py` | Modify | Add tags to schema, filter functions |
| `backend/main.py` | Modify | Add tags endpoints |
| `frontend/src/api.js` | Modify | Add tags API functions |
| `frontend/src/components/TagEditor.jsx` | Create | Tag editing UI |
| `frontend/src/components/Sidebar.jsx` | Modify | Add tag filter |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 5
**New Concepts**: Metadata management, faceted search, tag UX patterns

---

## 7. Reasoning Model Support

### Overview
Special handling for reasoning models (like o1, o3, DeepSeek-R1) that have separate "thinking" and "response" phases. Display the model's reasoning process alongside its answer.

### Why It's Valuable
- **For users**: See how advanced models think through problems
- **For learning**: Understand different LLM architectures
- **For transparency**: Understand why a model reached its conclusion

### What You'll Learn
- Different LLM response formats
- Conditional UI rendering
- The `reasoning_details` field in OpenRouter responses
- How chain-of-thought differs from regular responses

### Implementation Approach

#### Backend Changes

The current `query_model()` already extracts `reasoning_details`:

```python
return {
    'content': message.get('content'),
    'reasoning_details': message.get('reasoning_details')  # Already captured!
}
```

**Update Stage 1 response format in `backend/council.py`**
```python
async def stage1_collect_responses(question: str, ...) -> List[Dict[str, Any]]:
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    result = []
    for model, response in responses.items():
        if response:
            result.append({
                "model": model,
                "response": response['content'],
                "reasoning": response.get('reasoning_details')  # Include if present
            })

    return result
```

#### Frontend Changes

**Modify `frontend/src/components/Stage1.jsx`**
```jsx
function ModelResponse({ response }) {
    const [showReasoning, setShowReasoning] = useState(false);

    return (
        <div className="model-response">
            {/* Show reasoning toggle if available */}
            {response.reasoning && (
                <div className="reasoning-section">
                    <button
                        className="reasoning-toggle"
                        onClick={() => setShowReasoning(!showReasoning)}
                    >
                        {showReasoning ? '▼ Hide Reasoning' : '▶ Show Reasoning'}
                    </button>

                    {showReasoning && (
                        <div className="reasoning-content">
                            <h4>Model's Reasoning Process</h4>
                            <div className="reasoning-text">
                                {response.reasoning}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Main response */}
            <div className="markdown-content">
                <ReactMarkdown>{response.response}</ReactMarkdown>
            </div>
        </div>
    );
}
```

**Add styling for reasoning section**
```css
/* In Stage1.css */
.reasoning-section {
    margin-bottom: 16px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
}

.reasoning-toggle {
    width: 100%;
    padding: 12px;
    background: #f5f5f5;
    border: none;
    text-align: left;
    cursor: pointer;
    font-weight: 500;
}

.reasoning-content {
    padding: 16px;
    background: #fafafa;
    border-top: 1px solid #e0e0e0;
}

.reasoning-text {
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
    color: #666;
}
```

### Reasoning Model Examples

Models that support reasoning output:
- `openai/o1-preview` - Has separate thinking phase
- `openai/o1-mini` - Smaller reasoning model
- `deepseek/deepseek-r1` - Shows chain-of-thought

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/council.py` | Modify | Include reasoning_details in response |
| `frontend/src/components/Stage1.jsx` | Modify | Display reasoning with toggle |
| `frontend/src/components/Stage1.css` | Modify | Style reasoning section |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 3
**New Concepts**: Reasoning models, conditional rendering, collapsible UI

---

## 8. Cost Tracking

### Overview
Calculate and display the token cost for each query using OpenRouter's pricing data.

### Why It's Valuable
- **For users**: Budget awareness, understand API costs
- **For learning**: How token-based pricing works
- **For optimization**: Identify expensive queries, compare model costs

### What You'll Learn
- Token counting concepts
- API pricing models
- Real-time cost calculation
- Usage tracking and display

### Implementation Approach

#### Backend Changes

OpenRouter returns usage data in responses. Capture it:

**Modify `backend/openrouter.py`**
```python
async def query_model(model: str, messages: List[Dict[str, str]], timeout: float = 120.0):
    # ... existing code ...

    data = response.json()
    message = data['choices'][0]['message']
    usage = data.get('usage', {})

    return {
        'content': message.get('content'),
        'reasoning_details': message.get('reasoning_details'),
        'usage': {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
        }
    }
```

**Create `backend/pricing.py`**
```python
# Pricing per 1M tokens (as of 2024 - update periodically)
MODEL_PRICING = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "google/gemini-pro-1.5": {"input": 1.25, "output": 5.00},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
    "x-ai/grok-beta": {"input": 5.00, "output": 15.00},
    # Add more as needed
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a model query."""
    pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 1.0})

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost

def calculate_query_cost(responses: List[Dict]) -> Dict:
    """Calculate total cost for a full council query."""
    total_cost = 0.0
    breakdown = []

    for resp in responses:
        usage = resp.get('usage', {})
        model = resp.get('model', 'unknown')
        cost = calculate_cost(
            model,
            usage.get('prompt_tokens', 0),
            usage.get('completion_tokens', 0)
        )
        total_cost += cost
        breakdown.append({
            'model': model,
            'tokens': usage.get('total_tokens', 0),
            'cost': cost
        })

    return {
        'total_cost': total_cost,
        'breakdown': breakdown
    }
```

**Include cost in API response**
```python
# In main.py streaming endpoint
yield f"data: {json.dumps({
    'type': 'cost_update',
    'data': calculate_query_cost(all_responses)
})}\n\n"
```

#### Frontend Changes

**Create `frontend/src/components/CostDisplay.jsx`**
```jsx
export function CostDisplay({ cost }) {
    if (!cost) return null;

    return (
        <div className="cost-display">
            <div className="total-cost">
                Query Cost: ${cost.total_cost.toFixed(4)}
            </div>

            <details>
                <summary>Cost Breakdown</summary>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Tokens</th>
                            <th>Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cost.breakdown.map((item, i) => (
                            <tr key={i}>
                                <td>{item.model}</td>
                                <td>{item.tokens.toLocaleString()}</td>
                                <td>${item.cost.toFixed(4)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </details>
        </div>
    );
}
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/openrouter.py` | Modify | Capture usage data |
| `backend/pricing.py` | Create | Pricing data and calculation |
| `backend/main.py` | Modify | Include cost in response |
| `frontend/src/components/CostDisplay.jsx` | Create | Cost display UI |
| `frontend/src/App.jsx` | Modify | Show cost after query |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 5
**New Concepts**: Token pricing, usage tracking, cost optimization

---

## 9. Debate Mode

### Overview
Instead of independent answers, have models respond to each other in rounds. Model A answers → Model B critiques → Model A rebuts → Synthesis.

### Why It's Valuable
- **For users**: Deeper analysis through adversarial discussion
- **For learning**: Multi-turn conversation orchestration
- **For quality**: Surface edge cases and weak arguments

### What You'll Learn
- Multi-turn conversation management
- Context chaining between models
- Dialogue orchestration patterns
- Adversarial prompting

### Implementation Approach

#### New Flow

```
Round 1: Initial Positions
├── Model A: "I think X because..."
├── Model B: "I think Y because..."
└── Model C: "I think Z because..."

Round 2: Critiques
├── Model A critiques B and C
├── Model B critiques A and C
└── Model C critiques A and B

Round 3: Rebuttals
├── Model A responds to critiques
├── Model B responds to critiques
└── Model C responds to critiques

Final: Chairman synthesizes debate
```

#### Backend Changes

**Create `backend/debate.py`**
```python
from .openrouter import query_model, query_models_parallel
from .config_api import get_council_models, get_chairman_model

async def run_debate(question: str, rounds: int = 2) -> Dict:
    """
    Run a multi-round debate between models.

    Args:
        question: The topic to debate
        rounds: Number of critique/rebuttal rounds

    Returns:
        Dict with all rounds and final synthesis
    """
    models = get_council_models()
    debate_history = {"question": question, "rounds": []}

    # Round 1: Initial positions
    initial_prompt = [
        {"role": "user", "content": f"Question: {question}\n\nProvide your position on this question with supporting arguments."}
    ]

    initial_responses = await query_models_parallel(models, initial_prompt)
    round1 = {
        "type": "initial_positions",
        "responses": [
            {"model": m, "content": r["content"]}
            for m, r in initial_responses.items() if r
        ]
    }
    debate_history["rounds"].append(round1)

    # Subsequent rounds: Critiques and rebuttals
    for round_num in range(rounds):
        # Build critique prompt
        positions_text = "\n\n".join([
            f"**{r['model']}**: {r['content']}"
            for r in round1["responses"]
        ])

        critique_prompt = [
            {"role": "user", "content": f"""Question: {question}

Here are the current positions:

{positions_text}

Analyze the OTHER participants' arguments. Identify weaknesses, logical flaws, or missing considerations. Be constructive but thorough."""}
        ]

        critique_responses = await query_models_parallel(models, critique_prompt)
        critique_round = {
            "type": f"critiques_round_{round_num + 1}",
            "responses": [
                {"model": m, "content": r["content"]}
                for m, r in critique_responses.items() if r
            ]
        }
        debate_history["rounds"].append(critique_round)

        # Build rebuttal prompt
        critiques_text = "\n\n".join([
            f"**{r['model']}** critiqued: {r['content']}"
            for r in critique_round["responses"]
        ])

        rebuttal_prompt = [
            {"role": "user", "content": f"""The following critiques were raised:

{critiques_text}

Respond to the critiques of YOUR position. Address valid points and defend against unfair criticism."""}
        ]

        rebuttal_responses = await query_models_parallel(models, rebuttal_prompt)
        rebuttal_round = {
            "type": f"rebuttals_round_{round_num + 1}",
            "responses": [
                {"model": m, "content": r["content"]}
                for m, r in rebuttal_responses.items() if r
            ]
        }
        debate_history["rounds"].append(rebuttal_round)

    # Final synthesis
    full_debate = format_debate_for_synthesis(debate_history)
    synthesis_prompt = [
        {"role": "user", "content": f"""You are synthesizing a debate on: {question}

Here is the full debate:

{full_debate}

Provide a balanced synthesis that:
1. Identifies the strongest arguments from each side
2. Acknowledges points of genuine disagreement
3. Offers a nuanced conclusion that accounts for the debate"""}
    ]

    chairman = get_chairman_model()
    synthesis = await query_model(chairman, synthesis_prompt)
    debate_history["synthesis"] = {
        "model": chairman,
        "content": synthesis["content"]
    }

    return debate_history

def format_debate_for_synthesis(debate: Dict) -> str:
    """Format debate history for chairman prompt."""
    lines = []
    for round_data in debate["rounds"]:
        lines.append(f"## {round_data['type'].replace('_', ' ').title()}")
        for resp in round_data["responses"]:
            lines.append(f"**{resp['model']}**: {resp['content']}\n")
    return "\n".join(lines)
```

**Add endpoint to `backend/main.py`**
```python
from .debate import run_debate

@app.post("/api/conversations/{conversation_id}/debate")
async def start_debate(conversation_id: str, request: MessageRequest):
    """Run a debate instead of standard council deliberation."""
    result = await run_debate(request.content, rounds=2)

    # Save to conversation
    # ...

    return result
```

#### Frontend Changes

**Create `frontend/src/components/DebateView.jsx`**
```jsx
export function DebateView({ debate }) {
    return (
        <div className="debate-view">
            <h2>Debate: {debate.question}</h2>

            {debate.rounds.map((round, i) => (
                <div key={i} className="debate-round">
                    <h3>{round.type.replace(/_/g, ' ')}</h3>

                    <div className="round-responses">
                        {round.responses.map((resp, j) => (
                            <div key={j} className="debate-response">
                                <h4>{resp.model}</h4>
                                <ReactMarkdown>{resp.content}</ReactMarkdown>
                            </div>
                        ))}
                    </div>
                </div>
            ))}

            <div className="debate-synthesis">
                <h3>Synthesis</h3>
                <p><em>By {debate.synthesis.model}</em></p>
                <ReactMarkdown>{debate.synthesis.content}</ReactMarkdown>
            </div>
        </div>
    );
}
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/debate.py` | Create | Debate orchestration logic |
| `backend/main.py` | Modify | Add debate endpoint |
| `frontend/src/api.js` | Modify | Add debate API function |
| `frontend/src/components/DebateView.jsx` | Create | Debate display UI |
| `frontend/src/App.jsx` | Modify | Add debate mode toggle |

### Estimated Complexity
**Difficulty**: Hard
**Files Changed**: 5
**New Concepts**: Multi-turn orchestration, adversarial dialogue, context chaining

---

## 10. Confidence Voting

### Overview
Ask models to rate their confidence (1-10) alongside their answer. Display aggregate confidence scores.

### Why It's Valuable
- **For users**: Know when the council is uncertain vs. confident
- **For learning**: Uncertainty quantification in LLMs
- **For trust**: High disagreement = take answer with caution

### What You'll Learn
- Structured output parsing
- Uncertainty quantification
- Prompt engineering for metadata
- Data visualization

### Implementation Approach

#### Backend Changes

**Modify Stage 1 prompt in `backend/council.py`**
```python
async def stage1_collect_responses(question: str, ...) -> List[Dict[str, Any]]:
    prompt_content = f"""{question}

After your response, on a new line, provide your confidence level in this exact format:
CONFIDENCE: [1-10]

Where 1 = very uncertain, 10 = highly confident. Be honest about your certainty."""

    messages = [{"role": "user", "content": prompt_content}]
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    result = []
    for model, response in responses.items():
        if response:
            content, confidence = parse_confidence(response['content'])
            result.append({
                "model": model,
                "response": content,
                "confidence": confidence
            })

    return result

def parse_confidence(text: str) -> tuple[str, int]:
    """Extract confidence rating from response."""
    import re

    # Look for "CONFIDENCE: X" pattern
    match = re.search(r'CONFIDENCE:\s*(\d+)', text, re.IGNORECASE)

    if match:
        confidence = int(match.group(1))
        confidence = max(1, min(10, confidence))  # Clamp to 1-10
        # Remove confidence line from content
        content = re.sub(r'\n?CONFIDENCE:\s*\d+', '', text).strip()
        return content, confidence

    return text, None  # No confidence found

def calculate_confidence_stats(responses: List[Dict]) -> Dict:
    """Calculate aggregate confidence statistics."""
    confidences = [r['confidence'] for r in responses if r.get('confidence')]

    if not confidences:
        return None

    return {
        "average": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "spread": max(confidences) - min(confidences),
        "consensus": "high" if max(confidences) - min(confidences) <= 2 else "low"
    }
```

#### Frontend Changes

**Create `frontend/src/components/ConfidenceDisplay.jsx`**
```jsx
export function ConfidenceDisplay({ responses, stats }) {
    return (
        <div className="confidence-display">
            <h4>Confidence Levels</h4>

            {/* Visual bar for each model */}
            <div className="confidence-bars">
                {responses.map(resp => (
                    <div key={resp.model} className="confidence-bar-row">
                        <span className="model-name">{resp.model.split('/')[1]}</span>
                        <div className="confidence-bar">
                            <div
                                className="confidence-fill"
                                style={{
                                    width: `${(resp.confidence || 0) * 10}%`,
                                    backgroundColor: getConfidenceColor(resp.confidence)
                                }}
                            />
                        </div>
                        <span className="confidence-value">{resp.confidence || '?'}/10</span>
                    </div>
                ))}
            </div>

            {/* Aggregate stats */}
            {stats && (
                <div className="confidence-stats">
                    <p>
                        Average: <strong>{stats.average.toFixed(1)}</strong> |
                        Consensus: <strong>{stats.consensus}</strong>
                    </p>
                    {stats.consensus === 'low' && (
                        <p className="warning">
                            ⚠️ Models disagree significantly on confidence
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}

function getConfidenceColor(confidence) {
    if (confidence >= 8) return '#4caf50';  // Green
    if (confidence >= 5) return '#ff9800';  // Orange
    return '#f44336';  // Red
}
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/council.py` | Modify | Add confidence to prompts and parsing |
| `frontend/src/components/ConfidenceDisplay.jsx` | Create | Confidence visualization |
| `frontend/src/components/Stage1.jsx` | Modify | Include confidence display |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 3
**New Concepts**: Structured output parsing, uncertainty quantification, data visualization

---

## 11. Live Process Monitor

### Overview
Allow users to monitor and peek into the communications between models in real-time. A "verbosity knob" lets users control how much detail they see - from minimal (just final results) to debug mode (every prompt, response, and internal state).

### Why It's Valuable
- **For users**: Understand what's happening during the 30+ second deliberation process
- **For learning**: See exactly how multi-model orchestration works internally
- **For debugging**: Diagnose issues by inspecting actual prompts and responses
- **For transparency**: Build trust by showing the "black box" internals

### What You'll Learn
- Real-time event streaming architecture
- Logging and observability patterns
- UI state management for live updates
- Debug/verbose mode implementation
- SSE (Server-Sent Events) for granular updates

### Verbosity Levels

| Level | Name | What's Shown |
|-------|------|--------------|
| 0 | Minimal | Only final results (current behavior) |
| 1 | Normal | Stage transitions, timing, model names |
| 2 | Verbose | Prompts sent to models, response previews |
| 3 | Debug | Everything: full prompts, full responses, internal state, token counts |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend UI                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────────────────────────┐  │
│  │ Verbosity   │  │         Live Process Panel           │  │
│  │   Knob      │  │                                      │  │
│  │  ○ Minimal  │  │  [12:34:01] Stage 1 starting...      │  │
│  │  ○ Normal   │  │  [12:34:01] → Querying gpt-4o        │  │
│  │  ● Verbose  │  │  [12:34:02] → Querying claude-3.5    │  │
│  │  ○ Debug    │  │  [12:34:05] ← gpt-4o responded (2.1s)│  │
│  └─────────────┘  │  [12:34:06] ← claude-3.5 responded   │  │
│                   │  [12:34:06] Stage 1 complete (5.2s)  │  │
│                   │  [12:34:06] Stage 2 starting...      │  │
│                   │  ...                                 │  │
│                   └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ SSE Events
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Backend                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Process Logger                                       │   │
│  │                                                      │   │
│  │ log_event(level, type, data)                        │   │
│  │   → filters by verbosity                            │   │
│  │   → emits SSE event                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Council.py calls:                                         │
│    log_event(1, "stage_start", {stage: 1})                │
│    log_event(2, "prompt_sent", {model: "gpt-4o", prompt})  │
│    log_event(3, "full_response", {model, response, tokens})│
└─────────────────────────────────────────────────────────────┘
```

### Implementation Approach

#### Backend Changes

**Create `backend/process_logger.py`**
```python
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from enum import IntEnum

class VerbosityLevel(IntEnum):
    MINIMAL = 0   # Only final results
    NORMAL = 1    # Stage transitions, timing
    VERBOSE = 2   # Prompts, response previews
    DEBUG = 3     # Everything

class ProcessLogger:
    """
    Logs process events at different verbosity levels.
    Events below the current verbosity level are filtered out.
    """

    def __init__(self, verbosity: int = 1, emit_callback: Optional[Callable] = None):
        self.verbosity = verbosity
        self.emit_callback = emit_callback
        self.start_time = datetime.now()
        self.events = []

    def log(self, level: int, event_type: str, data: Dict[str, Any] = None):
        """
        Log an event if it meets the verbosity threshold.

        Args:
            level: Minimum verbosity level to show this event
            event_type: Type of event (stage_start, prompt_sent, etc.)
            data: Event data
        """
        if level > self.verbosity:
            return  # Filter out events above verbosity level

        event = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": int((datetime.now() - self.start_time).total_seconds() * 1000),
            "level": level,
            "type": event_type,
            "data": data or {}
        }

        self.events.append(event)

        if self.emit_callback:
            self.emit_callback(event)

    # Convenience methods for common events
    def stage_start(self, stage: int, description: str = ""):
        self.log(1, "stage_start", {"stage": stage, "description": description})

    def stage_complete(self, stage: int, duration_ms: int):
        self.log(1, "stage_complete", {"stage": stage, "duration_ms": duration_ms})

    def model_query_start(self, model: str):
        self.log(1, "model_query_start", {"model": model})

    def model_query_complete(self, model: str, duration_ms: int, success: bool):
        self.log(1, "model_query_complete", {
            "model": model,
            "duration_ms": duration_ms,
            "success": success
        })

    def prompt_sent(self, model: str, prompt: str):
        """Level 2: Show the prompt being sent"""
        # Truncate for verbose mode, full in debug
        preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        self.log(2, "prompt_sent", {"model": model, "prompt_preview": preview})
        self.log(3, "prompt_sent_full", {"model": model, "prompt": prompt})

    def response_received(self, model: str, response: str, tokens: int = None):
        """Level 2: Show response preview, Level 3: Full response"""
        preview = response[:300] + "..." if len(response) > 300 else response
        self.log(2, "response_preview", {
            "model": model,
            "preview": preview,
            "length": len(response)
        })
        self.log(3, "response_full", {
            "model": model,
            "response": response,
            "tokens": tokens
        })

    def anonymization_mapping(self, mapping: Dict[str, str]):
        """Level 2: Show the anonymization mapping"""
        self.log(2, "anonymization_mapping", {"mapping": mapping})

    def ranking_parsed(self, model: str, ranking: list):
        """Level 2: Show parsed ranking"""
        self.log(2, "ranking_parsed", {"model": model, "ranking": ranking})

    def debug_state(self, label: str, state: Any):
        """Level 3: Dump internal state for debugging"""
        self.log(3, "debug_state", {"label": label, "state": state})
```

**Modify `backend/council.py` to use logger**
```python
from .process_logger import ProcessLogger

async def stage1_collect_responses(
    question: str,
    conversation_history: List[Dict] = None,
    logger: ProcessLogger = None
) -> List[Dict[str, Any]]:
    """Collect responses with optional process logging."""

    logger = logger or ProcessLogger(verbosity=0)  # Default: no logging
    stage_start = datetime.now()

    logger.stage_start(1, "Collecting initial responses from council")

    messages = [{"role": "user", "content": question}]

    # Log the prompt being sent (verbosity 2+)
    logger.prompt_sent("all_models", question)

    responses = {}
    for model in COUNCIL_MODELS:
        logger.model_query_start(model)
        query_start = datetime.now()

        response = await query_model(model, messages)

        duration = int((datetime.now() - query_start).total_seconds() * 1000)
        logger.model_query_complete(model, duration, response is not None)

        if response:
            logger.response_received(model, response['content'])
            responses[model] = response

    stage_duration = int((datetime.now() - stage_start).total_seconds() * 1000)
    logger.stage_complete(1, stage_duration)

    # ... rest of function
```

**Modify streaming endpoint in `backend/main.py`**
```python
from .process_logger import ProcessLogger, VerbosityLevel

class MessageRequest(BaseModel):
    content: str
    verbosity: int = 1  # Default to normal verbosity

@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: MessageRequest):
    async def event_generator():
        events_queue = asyncio.Queue()

        def emit_event(event):
            events_queue.put_nowait(event)

        logger = ProcessLogger(
            verbosity=request.verbosity,
            emit_callback=emit_event
        )

        # Run council deliberation in background task
        async def run_deliberation():
            result = await run_council_with_logging(request.content, logger)
            await events_queue.put({"type": "deliberation_complete", "result": result})

        task = asyncio.create_task(run_deliberation())

        # Stream events as they occur
        while True:
            try:
                event = await asyncio.wait_for(events_queue.get(), timeout=0.1)

                if event.get("type") == "deliberation_complete":
                    # Send final results
                    yield f"data: {json.dumps({'type': 'complete', 'data': event['result']})}\n\n"
                    break
                else:
                    # Send process event
                    yield f"data: {json.dumps({'type': 'process_event', 'event': event})}\n\n"

            except asyncio.TimeoutError:
                # No event yet, continue waiting
                if task.done():
                    break
                continue

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

#### Frontend Changes

**Create `frontend/src/components/ProcessMonitor.jsx`**
```jsx
import { useState, useEffect, useRef } from 'react';
import './ProcessMonitor.css';

const VERBOSITY_LABELS = {
    0: 'Minimal',
    1: 'Normal',
    2: 'Verbose',
    3: 'Debug'
};

export function ProcessMonitor({ events, verbosity, onVerbosityChange }) {
    const logRef = useRef(null);
    const [expanded, setExpanded] = useState({});

    // Auto-scroll to bottom
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [events]);

    const toggleExpand = (index) => {
        setExpanded(prev => ({ ...prev, [index]: !prev[index] }));
    };

    const formatEvent = (event, index) => {
        const time = new Date(event.timestamp).toLocaleTimeString();
        const elapsed = `+${(event.elapsed_ms / 1000).toFixed(1)}s`;

        switch (event.type) {
            case 'stage_start':
                return (
                    <div className="event event-stage-start">
                        <span className="time">{time}</span>
                        <span className="elapsed">{elapsed}</span>
                        <span className="icon">▶</span>
                        <span className="message">
                            Stage {event.data.stage} starting: {event.data.description}
                        </span>
                    </div>
                );

            case 'stage_complete':
                return (
                    <div className="event event-stage-complete">
                        <span className="time">{time}</span>
                        <span className="elapsed">{elapsed}</span>
                        <span className="icon">✓</span>
                        <span className="message">
                            Stage {event.data.stage} complete ({event.data.duration_ms}ms)
                        </span>
                    </div>
                );

            case 'model_query_start':
                return (
                    <div className="event event-query">
                        <span className="time">{time}</span>
                        <span className="elapsed">{elapsed}</span>
                        <span className="icon">→</span>
                        <span className="message">
                            Querying <strong>{event.data.model}</strong>...
                        </span>
                    </div>
                );

            case 'model_query_complete':
                return (
                    <div className={`event event-response ${event.data.success ? '' : 'error'}`}>
                        <span className="time">{time}</span>
                        <span className="elapsed">{elapsed}</span>
                        <span className="icon">←</span>
                        <span className="message">
                            <strong>{event.data.model}</strong>
                            {event.data.success
                                ? ` responded (${event.data.duration_ms}ms)`
                                : ' failed'}
                        </span>
                    </div>
                );

            case 'prompt_sent':
                return (
                    <div className="event event-prompt expandable" onClick={() => toggleExpand(index)}>
                        <span className="time">{time}</span>
                        <span className="icon">{expanded[index] ? '▼' : '▶'}</span>
                        <span className="message">
                            Prompt to <strong>{event.data.model}</strong>
                        </span>
                        {expanded[index] && (
                            <pre className="expanded-content">{event.data.prompt_preview}</pre>
                        )}
                    </div>
                );

            case 'response_preview':
                return (
                    <div className="event event-response-preview expandable" onClick={() => toggleExpand(index)}>
                        <span className="time">{time}</span>
                        <span className="icon">{expanded[index] ? '▼' : '▶'}</span>
                        <span className="message">
                            Response from <strong>{event.data.model}</strong> ({event.data.length} chars)
                        </span>
                        {expanded[index] && (
                            <pre className="expanded-content">{event.data.preview}</pre>
                        )}
                    </div>
                );

            case 'anonymization_mapping':
                return (
                    <div className="event event-mapping">
                        <span className="time">{time}</span>
                        <span className="icon">🔀</span>
                        <span className="message">
                            Anonymization: {Object.entries(event.data.mapping).map(([label, model]) =>
                                `${label}=${model.split('/')[1]}`
                            ).join(', ')}
                        </span>
                    </div>
                );

            case 'debug_state':
                return (
                    <div className="event event-debug expandable" onClick={() => toggleExpand(index)}>
                        <span className="time">{time}</span>
                        <span className="icon">🔧</span>
                        <span className="message">{event.data.label}</span>
                        {expanded[index] && (
                            <pre className="expanded-content">
                                {JSON.stringify(event.data.state, null, 2)}
                            </pre>
                        )}
                    </div>
                );

            default:
                return (
                    <div className="event">
                        <span className="time">{time}</span>
                        <span className="message">{event.type}: {JSON.stringify(event.data)}</span>
                    </div>
                );
        }
    };

    return (
        <div className="process-monitor">
            <div className="monitor-header">
                <h3>Process Monitor</h3>
                <div className="verbosity-control">
                    <label>Verbosity:</label>
                    <select
                        value={verbosity}
                        onChange={(e) => onVerbosityChange(parseInt(e.target.value))}
                    >
                        {Object.entries(VERBOSITY_LABELS).map(([level, label]) => (
                            <option key={level} value={level}>{label}</option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="event-log" ref={logRef}>
                {events.length === 0 ? (
                    <div className="empty-state">
                        Waiting for activity...
                    </div>
                ) : (
                    events.map((event, i) => (
                        <div key={i}>{formatEvent(event, i)}</div>
                    ))
                )}
            </div>

            <div className="monitor-footer">
                <span>{events.length} events</span>
                <button onClick={() => setExpanded({})}>Collapse All</button>
            </div>
        </div>
    );
}
```

**Create `frontend/src/components/ProcessMonitor.css`**
```css
.process-monitor {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    height: 300px;
}

.monitor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: #2d2d2d;
    border-bottom: 1px solid #404040;
}

.monitor-header h3 {
    margin: 0;
    font-size: 13px;
    color: #fff;
}

.verbosity-control {
    display: flex;
    align-items: center;
    gap: 8px;
}

.verbosity-control select {
    background: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555;
    padding: 4px 8px;
    border-radius: 4px;
}

.event-log {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.event {
    padding: 4px 8px;
    margin: 2px 0;
    border-radius: 4px;
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 8px;
}

.event .time {
    color: #888;
    min-width: 70px;
}

.event .elapsed {
    color: #666;
    min-width: 50px;
}

.event .icon {
    min-width: 20px;
}

.event .message {
    flex: 1;
}

.event strong {
    color: #569cd6;
}

.event-stage-start {
    background: #264f78;
    color: #9cdcfe;
}

.event-stage-complete {
    background: #2d5a2d;
    color: #98c379;
}

.event-query {
    color: #ce9178;
}

.event-response.error {
    background: #5a1d1d;
    color: #f48771;
}

.event-prompt,
.event-response-preview {
    background: #2d2d2d;
}

.event-mapping {
    background: #4a3c2d;
    color: #dcdcaa;
}

.event-debug {
    background: #3c2d4a;
    color: #c586c0;
}

.expandable {
    cursor: pointer;
}

.expandable:hover {
    background: #383838;
}

.expanded-content {
    width: 100%;
    margin-top: 8px;
    padding: 8px;
    background: #1a1a1a;
    border-radius: 4px;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 11px;
    max-height: 200px;
    overflow-y: auto;
}

.monitor-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    background: #2d2d2d;
    border-top: 1px solid #404040;
    font-size: 11px;
    color: #888;
}

.monitor-footer button {
    background: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555;
    padding: 2px 8px;
    border-radius: 4px;
    cursor: pointer;
}

.empty-state {
    color: #666;
    text-align: center;
    padding: 40px;
}
```

**Integrate into App.jsx**
```jsx
import { ProcessMonitor } from './components/ProcessMonitor';

function App() {
    const [verbosity, setVerbosity] = useState(
        parseInt(localStorage.getItem('verbosity') || '1')
    );
    const [processEvents, setProcessEvents] = useState([]);

    const handleVerbosityChange = (level) => {
        setVerbosity(level);
        localStorage.setItem('verbosity', level.toString());
    };

    const handleSendMessage = async (content) => {
        setProcessEvents([]); // Clear previous events

        // Pass verbosity to API
        for await (const event of streamMessage(conversationId, content, verbosity)) {
            if (event.type === 'process_event') {
                setProcessEvents(prev => [...prev, event.event]);
            } else if (event.type === 'complete') {
                // Handle completion
            }
        }
    };

    return (
        <div className="app">
            {/* Main chat interface */}
            <div className="main-content">
                {/* ... existing chat components ... */}
            </div>

            {/* Process monitor panel */}
            {verbosity > 0 && (
                <ProcessMonitor
                    events={processEvents}
                    verbosity={verbosity}
                    onVerbosityChange={handleVerbosityChange}
                />
            )}
        </div>
    );
}
```

### Event Types Reference

| Event Type | Level | Description |
|------------|-------|-------------|
| `stage_start` | 1 | A stage is beginning |
| `stage_complete` | 1 | A stage has finished |
| `model_query_start` | 1 | Starting to query a model |
| `model_query_complete` | 1 | Model query finished |
| `prompt_sent` | 2 | The prompt being sent (preview) |
| `prompt_sent_full` | 3 | Full prompt text |
| `response_preview` | 2 | Response preview (truncated) |
| `response_full` | 3 | Full response with tokens |
| `anonymization_mapping` | 2 | Label-to-model mapping |
| `ranking_parsed` | 2 | Parsed ranking result |
| `debug_state` | 3 | Internal state dump |

### Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `backend/process_logger.py` | Create | Process logging infrastructure |
| `backend/council.py` | Modify | Add logging calls throughout |
| `backend/main.py` | Modify | Accept verbosity, emit events |
| `frontend/src/api.js` | Modify | Pass verbosity to API |
| `frontend/src/components/ProcessMonitor.jsx` | Create | Live process UI |
| `frontend/src/components/ProcessMonitor.css` | Create | Monitor styling |
| `frontend/src/App.jsx` | Modify | Integrate monitor, manage state |

### Estimated Complexity
**Difficulty**: Medium-Hard
**Files Changed**: 7
**New Concepts**: Real-time event streaming, observability, debug modes, SSE granular updates

### Use Cases

1. **Learning**: Watch how the 3-stage process works in real-time
2. **Debugging**: See exactly what prompts caused unexpected responses
3. **Performance**: Identify slow models or stages
4. **Demos**: Show stakeholders what's happening "under the hood"
5. **Development**: Test prompt changes and see immediate effects

### Notes

- Verbosity preference is persisted in localStorage
- Debug mode (level 3) can generate a lot of data - use sparingly
- Consider adding a "Download Log" button for sharing debug sessions
- The monitor panel can be collapsed/hidden when not needed

---

# Part 2: Orchestration Layer Features

These features focus on the core coordination logic - how models are selected, combined, and managed. They represent more advanced patterns used in production AI systems.

---

## 12. Dynamic Model Routing

### Overview
Analyze the question first using a classifier, then route to appropriate specialist models instead of always using the full council.

### Why It's Valuable
- **For users**: Faster responses, better answers from specialists
- **For learning**: Intent classification, conditional orchestration
- **For cost**: Use expensive models only when needed

### What You'll Learn
- Intent/topic classification
- Conditional orchestration flows
- Meta-prompting (using LLMs to decide which LLMs to use)
- Specialist vs. generalist model tradeoffs

### Architecture

```
User Question
     ↓
┌─────────────────────────┐
│ Classifier Model        │
│ (fast, cheap)           │
│ "What type of question  │
│  is this?"              │
└─────────────────────────┘
     ↓
     ├── "coding" ────────▶ [GPT-4o, Claude, DeepSeek-Coder]
     ├── "writing" ───────▶ [Claude, GPT-4o, Gemini]
     ├── "math" ──────────▶ [GPT-4o, Claude, Wolfram]
     ├── "research" ──────▶ [Perplexity, GPT-4o, Claude]
     └── "general" ───────▶ [Full Council]
```

### Implementation Approach

**Create `backend/router.py`**
```python
from .openrouter import query_model

# Model pools for different question types
MODEL_POOLS = {
    "coding": [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-coder",
    ],
    "writing": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-pro-1.5",
    ],
    "math": [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "openai/o1-mini",
    ],
    "research": [
        "perplexity/llama-3.1-sonar-large-128k-online",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
    ],
    "general": None,  # Use default council
}

CLASSIFIER_MODEL = "openai/gpt-4o-mini"  # Fast and cheap

async def classify_question(question: str) -> str:
    """
    Classify a question into a category.

    Returns:
        One of: "coding", "writing", "math", "research", "general"
    """
    classification_prompt = [
        {"role": "system", "content": """You are a question classifier.
Classify the user's question into exactly ONE of these categories:
- coding: Programming, software development, debugging, code review
- writing: Essays, stories, editing, grammar, style
- math: Mathematics, statistics, calculations, proofs
- research: Current events, factual questions, citations needed
- general: Everything else

Respond with ONLY the category name, nothing else."""},
        {"role": "user", "content": question}
    ]

    response = await query_model(CLASSIFIER_MODEL, classification_prompt, timeout=10.0)

    if response:
        category = response['content'].strip().lower()
        if category in MODEL_POOLS:
            return category

    return "general"

async def get_models_for_question(question: str) -> list[str]:
    """
    Determine which models to use for a question.

    Returns:
        List of model identifiers to query
    """
    category = await classify_question(question)

    pool = MODEL_POOLS.get(category)
    if pool:
        return pool

    # Fall back to default council
    from .config_api import get_council_models
    return get_council_models()
```

**Modify `backend/council.py`**
```python
from .router import get_models_for_question

async def stage1_collect_responses(question: str, use_routing: bool = True, ...):
    """
    Collect responses from council models.

    Args:
        question: The user's question
        use_routing: If True, dynamically select models based on question type
    """
    if use_routing:
        models = await get_models_for_question(question)
    else:
        models = get_council_models()

    responses = await query_models_parallel(models, messages)
    # ... rest of function
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/router.py` | Create | Question classification and routing |
| `backend/council.py` | Modify | Use router for model selection |
| `backend/main.py` | Modify | Add option to enable/disable routing |

### Estimated Complexity
**Difficulty**: Hard
**Files Changed**: 3
**New Concepts**: Intent classification, conditional orchestration, meta-prompting

---

## 13. Weighted Consensus Voting

### Overview
Models that historically rank higher get more voting power in Stage 2 rankings. The system learns which models to trust more.

### Why It's Valuable
- **For users**: Better final rankings based on track record
- **For learning**: Reputation systems, adaptive algorithms
- **For quality**: System improves over time

### What You'll Learn
- Reputation/trust scoring
- Adaptive weighting algorithms
- Feedback loops in AI systems
- Statistical aggregation methods

### Implementation Approach

**Create `backend/weights.py`**
```python
import json
import os

WEIGHTS_FILE = "data/model_weights.json"

def load_weights() -> dict[str, float]:
    """Load model weights from file."""
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_weights(weights: dict[str, float]):
    """Save model weights to file."""
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f, indent=2)

def get_model_weight(model: str) -> float:
    """Get weight for a model (default 1.0)."""
    weights = load_weights()
    return weights.get(model, 1.0)

def update_weights_from_rankings(aggregate_rankings: list[dict]):
    """
    Update model weights based on ranking results.

    Models ranked higher get weight increases, lower get decreases.
    Uses exponential moving average for stability.
    """
    weights = load_weights()
    learning_rate = 0.1

    for i, ranking in enumerate(aggregate_rankings):
        model = ranking['model']
        # Position 1 = best, position N = worst
        # Convert to score: 1st place gets highest score
        position = i + 1
        total = len(aggregate_rankings)
        score = (total - position + 1) / total  # 1.0 for 1st, lower for worse

        current_weight = weights.get(model, 1.0)
        new_weight = current_weight * (1 - learning_rate) + score * learning_rate
        weights[model] = new_weight

    save_weights(weights)

def calculate_weighted_aggregate(rankings: list[dict], label_to_model: dict) -> list[dict]:
    """
    Calculate aggregate rankings with model weights.

    Args:
        rankings: List of ranking dicts from Stage 2
        label_to_model: Mapping from "Response A" to model identifier

    Returns:
        Sorted list of {model, weighted_score, raw_score}
    """
    model_scores = {}

    for ranking_entry in rankings:
        ranker = ranking_entry['model']
        ranker_weight = get_model_weight(ranker)

        parsed = ranking_entry.get('parsed_ranking', [])
        for position, label in enumerate(parsed, 1):
            model = label_to_model.get(label)
            if model:
                if model not in model_scores:
                    model_scores[model] = {'weighted_sum': 0, 'raw_sum': 0, 'count': 0}

                # Lower position = better, so invert for scoring
                raw_score = len(parsed) - position + 1
                weighted_score = raw_score * ranker_weight

                model_scores[model]['weighted_sum'] += weighted_score
                model_scores[model]['raw_sum'] += raw_score
                model_scores[model]['count'] += 1

    # Convert to list and sort by weighted score
    result = []
    for model, scores in model_scores.items():
        result.append({
            'model': model,
            'weighted_score': scores['weighted_sum'] / scores['count'] if scores['count'] else 0,
            'raw_score': scores['raw_sum'] / scores['count'] if scores['count'] else 0,
            'vote_count': scores['count']
        })

    result.sort(key=lambda x: x['weighted_score'], reverse=True)
    return result
```

**Integrate into `backend/council.py`**
```python
from .weights import calculate_weighted_aggregate, update_weights_from_rankings

def calculate_aggregate_rankings(rankings, label_to_model, use_weights=True):
    """Calculate aggregate with optional weighting."""
    if use_weights:
        return calculate_weighted_aggregate(rankings, label_to_model)
    else:
        # Original unweighted calculation
        # ...
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/weights.py` | Create | Weight management and calculation |
| `backend/council.py` | Modify | Use weighted aggregation |
| `backend/main.py` | Modify | Update weights after each query |

### Estimated Complexity
**Difficulty**: Easy
**Files Changed**: 3
**New Concepts**: Reputation systems, adaptive algorithms, EMA

---

## 14. Early Consensus Exit

### Overview
If all models strongly agree in Stage 2 rankings, skip the chairman synthesis and return the top-ranked response directly.

### Why It's Valuable
- **For users**: Faster responses when agreement is clear
- **For learning**: Consensus detection algorithms
- **For cost**: Skip unnecessary API call to chairman

### What You'll Learn
- Consensus detection algorithms
- Threshold tuning
- Optimization through short-circuiting
- Statistical agreement measures

### Implementation Approach

**Add to `backend/council.py`**
```python
def check_consensus(rankings: list[dict], label_to_model: dict, threshold: float = 0.8) -> dict | None:
    """
    Check if there's strong consensus on the top response.

    Args:
        rankings: Stage 2 ranking results
        label_to_model: Label to model mapping
        threshold: Required agreement level (0.0 to 1.0)

    Returns:
        Winner dict if consensus, None otherwise
    """
    first_place_votes = {}
    total_voters = len(rankings)

    for ranking_entry in rankings:
        parsed = ranking_entry.get('parsed_ranking', [])
        if parsed:
            first_choice = parsed[0]  # "Response A", etc.
            first_place_votes[first_choice] = first_place_votes.get(first_choice, 0) + 1

    # Check if any response has enough first-place votes
    for label, votes in first_place_votes.items():
        agreement = votes / total_voters
        if agreement >= threshold:
            return {
                'label': label,
                'model': label_to_model.get(label),
                'agreement': agreement,
                'votes': votes,
                'total': total_voters
            }

    return None

async def run_council_deliberation(question: str, skip_synthesis_on_consensus: bool = True):
    """
    Run the full council deliberation with optional early exit.
    """
    # Stage 1
    stage1_responses = await stage1_collect_responses(question)

    # Stage 2
    rankings, label_to_model = await stage2_collect_rankings(question, stage1_responses)

    # Check for consensus
    if skip_synthesis_on_consensus:
        consensus = check_consensus(rankings, label_to_model)
        if consensus:
            # Find the winning response from Stage 1
            winner_response = next(
                (r for r in stage1_responses if r['model'] == consensus['model']),
                None
            )
            return {
                'stage1': stage1_responses,
                'stage2': rankings,
                'stage3': {
                    'model': 'consensus',
                    'response': winner_response['response'] if winner_response else '',
                    'note': f"Consensus reached ({consensus['agreement']:.0%} agreement). Skipped chairman synthesis."
                },
                'metadata': {
                    'label_to_model': label_to_model,
                    'consensus': consensus,
                    'skipped_synthesis': True
                }
            }

    # Stage 3 (no consensus, run normally)
    stage3_response = await stage3_synthesize_final(question, stage1_responses, rankings)

    return {
        'stage1': stage1_responses,
        'stage2': rankings,
        'stage3': stage3_response,
        'metadata': {
            'label_to_model': label_to_model,
            'skipped_synthesis': False
        }
    }
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/council.py` | Modify | Add consensus check and early exit |
| `frontend/src/components/Stage3.jsx` | Modify | Show consensus indicator |

### Estimated Complexity
**Difficulty**: Easy
**Files Changed**: 2
**New Concepts**: Consensus algorithms, short-circuit optimization

---

## 15. Iterative Refinement Loop

### Overview
Chairman drafts an answer, council critiques it, chairman revises. Repeat until quality threshold or max iterations.

### Why It's Valuable
- **For users**: Higher quality through revision cycles
- **For learning**: Iterative prompting, convergence detection
- **For quality**: Catch and fix issues through feedback

### What You'll Learn
- Iterative refinement patterns
- Convergence/termination criteria
- Quality assessment prompts
- Feedback incorporation

### Architecture

```
Stage 3: Chairman Draft v1
          ↓
Stage 4: Council Critique
          ↓
Stage 5: Chairman Draft v2
          ↓
Stage 6: Council Critique
          ↓
       (check: significant changes?)
          ↓
Stage 7: Final Answer
```

### Implementation Approach

**Create `backend/refinement.py`**
```python
from .openrouter import query_model, query_models_parallel
from .config_api import get_council_models, get_chairman_model

async def refine_response(
    question: str,
    initial_draft: str,
    stage1_responses: list,
    max_iterations: int = 2
) -> dict:
    """
    Iteratively refine a response through council feedback.

    Returns:
        Dict with all iterations and final response
    """
    iterations = []
    current_draft = initial_draft

    for i in range(max_iterations):
        # Get council critique
        critique_prompt = [
            {"role": "user", "content": f"""Original question: {question}

Current answer draft:
{current_draft}

Review this answer and provide constructive criticism:
1. What's missing or incomplete?
2. What could be explained better?
3. Are there any errors or inaccuracies?
4. What would make this answer more helpful?

Be specific and actionable."""}
        ]

        critiques = await query_models_parallel(get_council_models(), critique_prompt)

        critique_list = [
            {"model": m, "critique": r['content']}
            for m, r in critiques.items() if r
        ]

        # Check if critiques are substantive
        if not has_substantive_critiques(critique_list):
            iterations.append({
                "iteration": i + 1,
                "draft": current_draft,
                "critiques": critique_list,
                "stopped": "No substantive critiques - answer is good"
            })
            break

        # Chairman incorporates feedback
        feedback_text = "\n\n".join([
            f"**{c['model']}**: {c['critique']}"
            for c in critique_list
        ])

        revision_prompt = [
            {"role": "user", "content": f"""Original question: {question}

Your previous answer:
{current_draft}

Council feedback:
{feedback_text}

Revise your answer to address the valid critiques. Keep what's good, fix what's problematic."""}
        ]

        revision = await query_model(get_chairman_model(), revision_prompt)
        new_draft = revision['content'] if revision else current_draft

        iterations.append({
            "iteration": i + 1,
            "draft": current_draft,
            "critiques": critique_list,
            "revision": new_draft
        })

        current_draft = new_draft

    return {
        "iterations": iterations,
        "final_response": current_draft,
        "total_iterations": len(iterations)
    }

def has_substantive_critiques(critiques: list) -> bool:
    """
    Check if critiques contain substantive feedback.

    Simple heuristic: critiques mentioning "no issues", "looks good", etc.
    are not substantive.
    """
    non_substantive_phrases = [
        "no issues", "looks good", "well done", "comprehensive",
        "nothing to add", "excellent", "no changes needed"
    ]

    substantive_count = 0
    for c in critiques:
        text = c['critique'].lower()
        is_substantive = not any(phrase in text for phrase in non_substantive_phrases)
        if is_substantive:
            substantive_count += 1

    # Need majority to have substantive critiques
    return substantive_count > len(critiques) / 2
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/refinement.py` | Create | Iterative refinement logic |
| `backend/council.py` | Modify | Integrate refinement as optional stage |
| `backend/main.py` | Modify | Add refinement option to API |
| `frontend/src/components/Refinement.jsx` | Create | Show iteration history |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 4
**New Concepts**: Iterative refinement, convergence detection, feedback loops

---

## 16. Parallel Sub-Question Decomposition

### Overview
Break complex questions into parts, route each to the council separately, then merge the answers.

### Why It's Valuable
- **For users**: Better handling of multi-part questions
- **For learning**: Query decomposition, map-reduce patterns
- **For quality**: Focused answers for each aspect

### What You'll Learn
- Question decomposition techniques
- Map-reduce orchestration
- Context merging strategies
- Parallel execution patterns

### Architecture

```
Complex Question
     ↓
┌─────────────────────────┐
│ Decomposer              │
│ "Split this into parts" │
└─────────────────────────┘
     ↓
┌────────┬────────┬────────┐
│ Sub-Q1 │ Sub-Q2 │ Sub-Q3 │
└────┬───┴────┬───┴────┬───┘
     ↓        ↓        ↓
┌────────┬────────┬────────┐
│Council │Council │Council │  (parallel)
│  for   │  for   │  for   │
│ Sub-Q1 │ Sub-Q2 │ Sub-Q3 │
└────┬───┴────┬───┴────┬───┘
     ↓        ↓        ↓
┌─────────────────────────┐
│ Merger                  │
│ "Combine into coherent  │
│  final answer"          │
└─────────────────────────┘
```

### Implementation Approach

**Create `backend/decompose.py`**
```python
from .openrouter import query_model
from .council import run_council_deliberation
import asyncio

DECOMPOSER_MODEL = "openai/gpt-4o-mini"

async def decompose_question(question: str) -> list[str]:
    """
    Break a complex question into sub-questions.

    Returns:
        List of sub-questions, or [original] if not decomposable
    """
    decompose_prompt = [
        {"role": "system", "content": """You are a question decomposer.
If the question has multiple distinct parts that could be answered separately, split it.
If it's a simple, focused question, return it as-is.

Output format:
- One question per line
- No numbering or bullets
- Each question should be self-contained"""},
        {"role": "user", "content": question}
    ]

    response = await query_model(DECOMPOSER_MODEL, decompose_prompt, timeout=15.0)

    if response:
        sub_questions = [q.strip() for q in response['content'].split('\n') if q.strip()]
        # Only decompose if we got 2-5 sub-questions
        if 2 <= len(sub_questions) <= 5:
            return sub_questions

    return [question]  # Return original if decomposition failed

async def run_decomposed_council(question: str) -> dict:
    """
    Run council deliberation on decomposed question parts.
    """
    # Step 1: Decompose
    sub_questions = await decompose_question(question)

    if len(sub_questions) == 1:
        # Not decomposed, run normal council
        result = await run_council_deliberation(question)
        result['decomposed'] = False
        return result

    # Step 2: Run council on each sub-question in parallel
    tasks = [run_council_deliberation(sq) for sq in sub_questions]
    sub_results = await asyncio.gather(*tasks)

    # Step 3: Merge results
    merged = await merge_results(question, sub_questions, sub_results)

    return {
        'decomposed': True,
        'original_question': question,
        'sub_questions': sub_questions,
        'sub_results': sub_results,
        'merged_response': merged
    }

async def merge_results(original: str, sub_questions: list, sub_results: list) -> str:
    """
    Merge sub-question answers into coherent final response.
    """
    # Build context from sub-answers
    parts = []
    for sq, result in zip(sub_questions, sub_results):
        answer = result.get('stage3', {}).get('response', '')
        parts.append(f"**{sq}**\n{answer}")

    parts_text = "\n\n---\n\n".join(parts)

    merge_prompt = [
        {"role": "user", "content": f"""Original question: {original}

The question was broken into parts and each was answered by a council:

{parts_text}

Synthesize these into a single, coherent response that:
1. Flows naturally without obvious section breaks
2. Eliminates redundancy between sections
3. Directly addresses the original question
4. Maintains all important details from sub-answers"""}
    ]

    response = await query_model(get_chairman_model(), merge_prompt)
    return response['content'] if response else parts_text
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/decompose.py` | Create | Question decomposition and merging |
| `backend/main.py` | Modify | Add decomposition endpoint/option |
| `frontend/src/components/DecomposedView.jsx` | Create | Show sub-questions and merge |

### Estimated Complexity
**Difficulty**: Hard
**Files Changed**: 3
**New Concepts**: Query decomposition, map-reduce, parallel orchestration

---

## 17. Adversarial Validation Stage

### Overview
Add a "devil's advocate" model that tries to find flaws in the chairman's synthesis before final output.

### Why It's Valuable
- **For users**: Catch errors before presenting the answer
- **For learning**: Adversarial prompting, self-correction
- **For quality**: More robust, defensible answers

### What You'll Learn
- Adversarial prompting techniques
- Self-correction patterns
- Quality gates in pipelines
- Error detection strategies

### Architecture

```
Stage 3: Chairman Synthesis
          ↓
Stage 4: Adversary Model
         "Find errors, inconsistencies,
          weak arguments, factual issues"
          ↓
     ┌────┴────┐
     ↓         ↓
 No issues   Issues found
     ↓         ↓
  Output    Stage 5: Chairman
            addresses criticisms
                ↓
             Output
```

### Implementation Approach

**Create `backend/adversary.py`**
```python
from .openrouter import query_model

ADVERSARY_MODEL = "anthropic/claude-3.5-sonnet"  # Good at critique

async def adversarial_review(
    question: str,
    synthesis: str,
    stage1_responses: list
) -> dict:
    """
    Have an adversary model review the synthesis for issues.

    Returns:
        Dict with issues found and severity assessment
    """
    # Prepare context
    original_responses = "\n\n".join([
        f"**{r['model']}**: {r['response']}"
        for r in stage1_responses
    ])

    adversary_prompt = [
        {"role": "system", "content": """You are a critical reviewer. Your job is to find problems.
Look for:
- Factual errors or unsupported claims
- Logical inconsistencies
- Important points from original responses that were omitted
- Misleading simplifications
- Unclear or ambiguous statements

Be thorough but fair. If the synthesis is good, say so."""},
        {"role": "user", "content": f"""Question: {question}

Original council responses:
{original_responses}

Synthesized answer to review:
{synthesis}

List any issues you find. For each issue:
1. What the problem is
2. Why it matters
3. How it could be fixed

If no significant issues, respond with "NO SIGNIFICANT ISSUES FOUND"."""}
    ]

    response = await query_model(ADVERSARY_MODEL, adversary_prompt)

    if response:
        content = response['content']
        has_issues = "NO SIGNIFICANT ISSUES FOUND" not in content.upper()

        return {
            'has_issues': has_issues,
            'review': content,
            'adversary_model': ADVERSARY_MODEL
        }

    return {'has_issues': False, 'review': 'Review failed', 'adversary_model': ADVERSARY_MODEL}

async def address_issues(
    question: str,
    original_synthesis: str,
    adversary_review: str
) -> str:
    """
    Have chairman address the adversary's concerns.
    """
    revision_prompt = [
        {"role": "user", "content": f"""Question: {question}

Your previous synthesis:
{original_synthesis}

A critical reviewer raised these concerns:
{adversary_review}

Revise your synthesis to address the valid concerns while maintaining accuracy.
If a concern is invalid, you may disregard it, but be sure the valid ones are addressed."""}
    ]

    response = await query_model(get_chairman_model(), revision_prompt)
    return response['content'] if response else original_synthesis
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/adversary.py` | Create | Adversarial review logic |
| `backend/council.py` | Modify | Add optional adversarial stage |
| `frontend/src/components/AdversaryReview.jsx` | Create | Show adversary feedback |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 3
**New Concepts**: Adversarial prompting, self-correction, quality gates

---

## 18. Confidence-Gated Escalation

### Overview
Start with cheap models; if confidence is low or disagreement is high, escalate to premium models.

### Why It's Valuable
- **For users**: Fast answers for easy questions, thorough answers for hard ones
- **For learning**: Tiered architectures, uncertainty handling
- **For cost**: Significant savings on routine queries

### What You'll Learn
- Tiered model architectures
- Uncertainty quantification
- Cost optimization strategies
- Escalation patterns

### Architecture

```
User Question
     ↓
┌─────────────────────────┐
│ Tier 1: Cheap Models    │
│ GPT-4o-mini, Haiku,     │
│ Gemini Flash            │
└─────────────────────────┘
     ↓
 Assess Confidence
     ↓
  ┌──┴──┐
High   Low
  ↓     ↓
Output  ┌─────────────────────────┐
        │ Tier 2: Premium Models  │
        │ GPT-4o, Claude 3.5,     │
        │ Gemini Pro              │
        └─────────────────────────┘
              ↓
           Output
```

### Implementation Approach

**Create `backend/escalation.py`**
```python
from .openrouter import query_models_parallel
from .council import stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final

TIER1_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "google/gemini-flash-1.5",
]

TIER2_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro-1.5",
]

CONFIDENCE_THRESHOLD = 6.0  # Average confidence below this triggers escalation
AGREEMENT_THRESHOLD = 0.5   # Agreement below this triggers escalation

async def run_tiered_council(question: str) -> dict:
    """
    Run council with automatic escalation based on confidence.
    """
    # Tier 1: Try with cheap models
    tier1_result = await run_council_with_models(question, TIER1_MODELS)

    # Assess confidence
    should_escalate = assess_escalation_need(tier1_result)

    if not should_escalate:
        tier1_result['tier'] = 1
        tier1_result['escalated'] = False
        return tier1_result

    # Tier 2: Escalate to premium models
    tier2_result = await run_council_with_models(question, TIER2_MODELS)
    tier2_result['tier'] = 2
    tier2_result['escalated'] = True
    tier2_result['escalation_reason'] = should_escalate
    tier2_result['tier1_preview'] = {
        'stage1': tier1_result['stage1'],
        'confidence': tier1_result.get('confidence_stats')
    }

    return tier2_result

def assess_escalation_need(result: dict) -> str | None:
    """
    Determine if escalation is needed based on Tier 1 results.

    Returns:
        Reason string if escalation needed, None otherwise
    """
    # Check confidence scores
    confidence_stats = result.get('confidence_stats', {})
    avg_confidence = confidence_stats.get('average', 10)

    if avg_confidence < CONFIDENCE_THRESHOLD:
        return f"Low confidence ({avg_confidence:.1f}/10)"

    # Check ranking agreement
    rankings = result.get('stage2', [])
    agreement = calculate_ranking_agreement(rankings)

    if agreement < AGREEMENT_THRESHOLD:
        return f"High disagreement (agreement: {agreement:.0%})"

    return None

def calculate_ranking_agreement(rankings: list) -> float:
    """
    Calculate how much the rankings agree.

    Returns:
        Float from 0.0 (complete disagreement) to 1.0 (complete agreement)
    """
    if len(rankings) < 2:
        return 1.0

    # Compare first-place votes
    first_places = [r['parsed_ranking'][0] for r in rankings if r.get('parsed_ranking')]
    if not first_places:
        return 0.5

    # Most common first place
    from collections import Counter
    counts = Counter(first_places)
    most_common_count = counts.most_common(1)[0][1]

    return most_common_count / len(first_places)
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/escalation.py` | Create | Tiered escalation logic |
| `backend/main.py` | Modify | Add escalation endpoint/option |
| `frontend/src/components/TierIndicator.jsx` | Create | Show which tier was used |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 3
**New Concepts**: Tiered architectures, uncertainty handling, cost optimization

---

## 19. Response Caching Layer

### Overview
Cache responses for semantically similar questions to avoid redundant API calls.

### Why It's Valuable
- **For users**: Instant responses for repeated topics
- **For learning**: Embeddings, vector databases, semantic similarity
- **For cost**: Significant savings on repeated queries

### What You'll Learn
- Text embeddings
- Vector similarity search
- Cache strategies
- Semantic matching vs. exact matching

### Architecture

```
New Question
     ↓
┌─────────────────────────┐
│ Generate Embedding      │
│ (question → vector)     │
└─────────────────────────┘
     ↓
┌─────────────────────────┐
│ Search Cache            │
│ "Find similar questions"│
└─────────────────────────┘
     ↓
  ┌──┴──┐
Found  Not Found
  ↓       ↓
Return   Run Council
Cached      ↓
         Cache Result
            ↓
         Return
```

### Implementation Approach

**Create `backend/cache.py`**
```python
import json
import os
import numpy as np
from .openrouter import query_model

CACHE_FILE = "data/response_cache.json"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.85

def load_cache() -> list[dict]:
    """Load cache from file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_cache(cache: list[dict]):
    """Save cache to file."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

async def get_embedding(text: str) -> list[float]:
    """
    Get embedding vector for text.

    Note: OpenRouter doesn't directly support embeddings.
    Options:
    1. Use OpenAI API directly for embeddings
    2. Use a local embedding model
    3. Use a simple hash-based approach for demo
    """
    # For production, use actual embedding API
    # This is a placeholder using a simple approach
    import hashlib

    # Create pseudo-embedding from hash (NOT for production!)
    hash_bytes = hashlib.sha256(text.lower().encode()).digest()
    return [b / 255.0 for b in hash_bytes]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def find_cached_response(question: str) -> dict | None:
    """
    Find a cached response for a similar question.

    Returns:
        Cached result dict if found, None otherwise
    """
    cache = load_cache()
    if not cache:
        return None

    question_embedding = await get_embedding(question)

    best_match = None
    best_similarity = 0

    for entry in cache:
        similarity = cosine_similarity(question_embedding, entry['embedding'])
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = entry

    if best_match and best_similarity >= SIMILARITY_THRESHOLD:
        return {
            'cached': True,
            'similarity': best_similarity,
            'original_question': best_match['question'],
            'result': best_match['result']
        }

    return None

async def cache_response(question: str, result: dict):
    """
    Add a response to the cache.
    """
    cache = load_cache()
    embedding = await get_embedding(question)

    cache.append({
        'question': question,
        'embedding': embedding,
        'result': result,
        'cached_at': datetime.now().isoformat()
    })

    # Limit cache size
    if len(cache) > 1000:
        cache = cache[-1000:]

    save_cache(cache)
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/cache.py` | Create | Caching logic with embeddings |
| `backend/main.py` | Modify | Check cache before running council |
| `requirements.txt` | Modify | Add numpy (if not present) |

### Estimated Complexity
**Difficulty**: Hard
**Files Changed**: 3
**New Concepts**: Embeddings, vector similarity, caching strategies

### Notes
- For production, use proper embedding models (OpenAI, Cohere, or local)
- Consider using a vector database (Pinecone, Weaviate, ChromaDB) for scale
- The placeholder implementation above is for demonstration only

---

## 20. Chain-of-Thought Orchestration

### Overview
Force models to show step-by-step reasoning, then have the council evaluate the *reasoning quality* not just the final answer.

### Why It's Valuable
- **For users**: See how models think, not just what they conclude
- **For learning**: Chain-of-thought prompting, reasoning evaluation
- **For quality**: Catch flawed logic even when answer looks right

### What You'll Learn
- Chain-of-thought prompting
- Reasoning evaluation criteria
- Structured output formats
- Process vs. outcome evaluation

### Implementation Approach

**Modify Stage 1 prompt in `backend/council.py`**
```python
async def stage1_collect_responses_cot(question: str, ...) -> List[Dict[str, Any]]:
    """
    Collect responses with explicit chain-of-thought reasoning.
    """
    cot_prompt = f"""Question: {question}

Please answer this question by showing your complete reasoning process:

1. First, identify what the question is really asking
2. List the key facts or considerations relevant to answering
3. Work through the logic step by step
4. Identify any assumptions you're making
5. State your conclusion

Format your response as:
## Understanding
[What the question is asking]

## Key Considerations
[Relevant facts, concepts, constraints]

## Reasoning
[Step-by-step logic]

## Assumptions
[Any assumptions made]

## Conclusion
[Your final answer]"""

    messages = [{"role": "user", "content": cot_prompt}]
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    result = []
    for model, response in responses.items():
        if response:
            parsed = parse_cot_response(response['content'])
            result.append({
                "model": model,
                "response": response['content'],
                "reasoning": parsed
            })

    return result

def parse_cot_response(text: str) -> dict:
    """Parse structured CoT response into components."""
    sections = {}
    current_section = None
    current_content = []

    for line in text.split('\n'):
        if line.startswith('## '):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].lower()
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections
```

**Modify Stage 2 evaluation prompt**
```python
async def stage2_evaluate_reasoning(question: str, stage1_responses: list) -> list:
    """
    Have models evaluate each other's reasoning, not just conclusions.
    """
    evaluation_prompt = f"""Question: {question}

Below are responses from different models, each showing their reasoning process.
Evaluate each response on:

1. **Reasoning Quality** (1-10): Is the logic sound? Are steps justified?
2. **Completeness** (1-10): Are all relevant factors considered?
3. **Assumptions** (1-10): Are assumptions reasonable and stated?
4. **Conclusion Validity** (1-10): Does the conclusion follow from the reasoning?

{format_responses_for_evaluation(stage1_responses)}

For each response, provide scores and brief justification.
Then rank them from best to worst reasoning (not just best answer).

FINAL RANKING:
1. Response X
2. Response Y
..."""

    # ... rest of evaluation logic
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/council.py` | Modify | Add CoT prompts and parsing |
| `frontend/src/components/ReasoningView.jsx` | Create | Display structured reasoning |
| `frontend/src/components/Stage1.jsx` | Modify | Show reasoning sections |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 3
**New Concepts**: Chain-of-thought, reasoning evaluation, structured prompts

---

## 21. Multi-Chairman Synthesis

### Overview
Multiple chairman models synthesize independently, then a "supreme chairman" picks the best or merges them.

### Why It's Valuable
- **For users**: More robust final answer
- **For learning**: Ensemble methods, meta-synthesis
- **For reliability**: No single point of failure in synthesis

### What You'll Learn
- Ensemble synthesis methods
- Meta-evaluation patterns
- Redundancy in AI systems
- Quality comparison techniques

### Architecture

```
Stage 2 Complete
     ↓
┌─────────┬─────────┬─────────┐
│Chairman │Chairman │Chairman │  (parallel)
│    A    │    B    │    C    │
│ (GPT)   │(Claude) │(Gemini) │
└────┬────┴────┬────┴────┬────┘
     ↓         ↓         ↓
  Synth A   Synth B   Synth C
     └─────────┼─────────┘
               ↓
┌─────────────────────────────┐
│ Supreme Chairman            │
│ "Which synthesis is best?   │
│  Or combine the best parts" │
└─────────────────────────────┘
               ↓
         Final Answer
```

### Implementation Approach

**Create `backend/multi_chairman.py`**
```python
from .openrouter import query_model, query_models_parallel

CHAIRMAN_POOL = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro-1.5",
]

SUPREME_CHAIRMAN = "anthropic/claude-3.5-sonnet"

async def multi_chairman_synthesis(
    question: str,
    stage1_responses: list,
    stage2_rankings: list
) -> dict:
    """
    Generate multiple syntheses and select/merge the best.
    """
    # Build synthesis prompt (same for all chairmen)
    synthesis_prompt = build_synthesis_prompt(question, stage1_responses, stage2_rankings)

    # Get syntheses from all chairmen in parallel
    messages = [{"role": "user", "content": synthesis_prompt}]
    syntheses = await query_models_parallel(CHAIRMAN_POOL, messages)

    synthesis_list = [
        {"model": m, "synthesis": r['content']}
        for m, r in syntheses.items() if r
    ]

    # Have supreme chairman evaluate and merge
    final = await supreme_synthesis(question, synthesis_list)

    return {
        "individual_syntheses": synthesis_list,
        "final_synthesis": final,
        "supreme_chairman": SUPREME_CHAIRMAN
    }

async def supreme_synthesis(question: str, syntheses: list) -> str:
    """
    Evaluate multiple syntheses and produce final answer.
    """
    syntheses_text = "\n\n---\n\n".join([
        f"**Synthesis by {s['model']}:**\n{s['synthesis']}"
        for s in syntheses
    ])

    supreme_prompt = [
        {"role": "user", "content": f"""Question: {question}

Three different AI models produced these synthesis answers:

{syntheses_text}

Your task:
1. Evaluate each synthesis for accuracy, completeness, and clarity
2. Identify the strengths of each
3. Produce a FINAL answer that:
   - Uses the best elements from each synthesis
   - Corrects any errors you notice
   - Provides the most helpful response to the original question

Your final answer:"""}
    ]

    response = await query_model(SUPREME_CHAIRMAN, supreme_prompt)
    return response['content'] if response else syntheses[0]['synthesis']

def build_synthesis_prompt(question: str, responses: list, rankings: list) -> str:
    """Build the standard synthesis prompt."""
    responses_text = "\n\n".join([
        f"**{r['model']}**: {r['response']}"
        for r in responses
    ])

    rankings_text = "\n\n".join([
        f"**{r['model']}** ranked: {', '.join(r.get('parsed_ranking', []))}"
        for r in rankings
    ])

    return f"""Question: {question}

Council Responses:
{responses_text}

Council Rankings:
{rankings_text}

Synthesize these into a single, authoritative answer that:
1. Incorporates the best insights from each response
2. Reflects the consensus shown in the rankings
3. Provides a clear, helpful answer to the original question"""
```

### Files to Modify/Create
| File | Action | Description |
|------|--------|-------------|
| `backend/multi_chairman.py` | Create | Multi-chairman synthesis logic |
| `backend/council.py` | Modify | Option to use multi-chairman |
| `frontend/src/components/MultiSynthesis.jsx` | Create | Show all syntheses |

### Estimated Complexity
**Difficulty**: Medium
**Files Changed**: 3
**New Concepts**: Ensemble methods, meta-synthesis, redundancy patterns

---

# Reference

## Complexity Summary

### General Features

| # | Feature | Difficulty | Files | Key Learning |
|---|---------|------------|-------|--------------|
| 1 | Performance Dashboard | Medium | 6 | Data visualization |
| 2 | Conversation Export | Easy | 2-3 | File generation |
| 3 | Custom System Prompts | Easy | 4 | Prompt engineering |
| 4 | Streaming Responses | Hard | 5 | Real-time systems |
| 5 | Model Configuration UI | Medium | 6 | Full-stack forms |
| 6 | Question Categories | Medium | 5 | Data modeling |
| 7 | Reasoning Model Support | Medium | 3 | LLM architectures |
| 8 | Cost Tracking | Medium | 5 | API pricing |
| 9 | Debate Mode | Hard | 5 | Multi-turn orchestration |
| 10 | Confidence Voting | Medium | 3 | Uncertainty quantification |
| 11 | Live Process Monitor | Medium-Hard | 7 | Observability, real-time events |

### Orchestration Features

| # | Feature | Difficulty | Files | Key Learning |
|---|---------|------------|-------|--------------|
| 12 | Dynamic Model Routing | Hard | 3 | Intent classification |
| 13 | Weighted Consensus | Easy | 3 | Adaptive algorithms |
| 14 | Early Consensus Exit | Easy | 2 | Optimization |
| 15 | Iterative Refinement | Medium | 4 | Feedback loops |
| 16 | Sub-Question Decomposition | Hard | 3 | Map-reduce patterns |
| 17 | Adversarial Validation | Medium | 3 | Self-correction |
| 18 | Confidence-Gated Escalation | Medium | 3 | Tiered architectures |
| 19 | Response Caching | Hard | 3 | Semantic search |
| 20 | Chain-of-Thought | Medium | 3 | Reasoning patterns |
| 21 | Multi-Chairman | Medium | 3 | Ensemble methods |

---

## Recommended Learning Paths

### Path 1: Beginner (Start Here)
Focus on understanding the basics and building confidence.

1. **Custom System Prompts** (#3) - Understand prompt engineering
2. **Conversation Export** (#2) - Simple file operations
3. **Early Consensus Exit** (#14) - Simple orchestration optimization
4. **Weighted Consensus** (#13) - Adaptive systems basics

### Path 2: Intermediate
Build on fundamentals with more complex patterns.

1. **Cost Tracking** (#8) - Understand token economics
2. **Confidence Voting** (#10) - Uncertainty quantification
3. **Live Process Monitor** (#11) - Observability and debugging
4. **Adversarial Validation** (#17) - Self-correction patterns
5. **Iterative Refinement** (#15) - Feedback loops
6. **Model Configuration UI** (#5) - Full-stack development

### Path 3: Advanced
Tackle complex orchestration and real-time systems.

1. **Streaming Responses** (#4) - Real-time architecture
2. **Live Process Monitor** (#11) - Real-time event streaming
3. **Dynamic Model Routing** (#12) - Intent classification
4. **Confidence-Gated Escalation** (#18) - Tiered systems
5. **Sub-Question Decomposition** (#16) - Map-reduce
6. **Response Caching** (#19) - Semantic search

### Path 4: Orchestration Deep Dive
Focus specifically on coordination patterns.

1. **Early Consensus Exit** (#14)
2. **Weighted Consensus** (#13)
3. **Adversarial Validation** (#17)
4. **Confidence-Gated Escalation** (#18)
5. **Dynamic Model Routing** (#12)
6. **Multi-Chairman Synthesis** (#21)

---

## Feature Dependencies

Some features build on others or work well together:

```
Cost Tracking (#8)
     └──▶ Confidence-Gated Escalation (#18)
              └──▶ Response Caching (#19)

Confidence Voting (#10)
     └──▶ Early Consensus Exit (#14)
     └──▶ Confidence-Gated Escalation (#18)

Custom System Prompts (#3)
     └──▶ Dynamic Model Routing (#12)

Model Configuration UI (#5)
     └──▶ Performance Dashboard (#1)
     └──▶ Weighted Consensus (#13)

Chain-of-Thought (#20)
     └──▶ Adversarial Validation (#17)
     └──▶ Iterative Refinement (#15)

Streaming Responses (#4)
     └──▶ Live Process Monitor (#11)
     └──▶ Debate Mode (#9)

Live Process Monitor (#11)
     └──▶ Cost Tracking (#8) - shows cost per event
     └──▶ Streaming Responses (#4) - enhanced real-time display
```

### Suggested Combinations

**"Production Ready" Stack:**
- Cost Tracking (#8)
- Live Process Monitor (#11)
- Confidence-Gated Escalation (#18)
- Response Caching (#19)
- Model Configuration UI (#5)

**"Quality Focus" Stack:**
- Chain-of-Thought (#20)
- Adversarial Validation (#17)
- Iterative Refinement (#15)
- Multi-Chairman (#21)

**"User Experience" Stack:**
- Streaming Responses (#4)
- Live Process Monitor (#11)
- Custom System Prompts (#3)
- Question Categories (#6)
- Conversation Export (#2)

---

## Getting Started

To implement any feature:

1. **Read the full description** in this document
2. **Review the files to modify** section
3. **Start with backend changes** - they're usually simpler to test
4. **Add frontend incrementally** - get API working first
5. **Test thoroughly** - use the curl examples and Swagger UI

For questions or clarification, refer to:
- [Backend Guide](./BACKEND_GUIDE.md) - How existing code works
- [Frontend Guide](./FRONTEND_GUIDE.md) - React patterns used
- [API Reference](./API_REFERENCE.md) - Endpoint documentation
- [Architecture](./ARCHITECTURE.md) - System design overview
