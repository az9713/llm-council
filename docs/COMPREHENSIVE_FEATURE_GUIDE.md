# LLM Council - Comprehensive Feature Implementation Guide

## Introduction

This document provides exhaustive documentation of all 21 features implemented in the LLM Council project. It is written for developers with **no prior experience** with Python, JavaScript, React, or web development. Every decision, file modification, and implementation detail is explained in plain language.

### What is LLM Council?

LLM Council is a web application that queries multiple AI language models (like GPT-4, Claude, Gemini) simultaneously about a user's question. Instead of getting one AI's opinion, you get several. The models then rank each other's answers (without knowing who said what), and a "chairman" model synthesizes the best final answer.

### How the Project is Structured

The project has two main parts:

1. **Backend (Python)** - The "server" that talks to AI models and processes data
   - Located in the `backend/` folder
   - Written in Python using FastAPI framework
   - Runs on port 8001 (localhost:8001)

2. **Frontend (JavaScript/React)** - The "user interface" you see in your browser
   - Located in the `frontend/src/` folder
   - Written in JavaScript using React framework
   - Runs on port 5173 (localhost:5173)

### File Types Explained

- `.py` files - Python code (backend logic)
- `.js` files - JavaScript code (frontend logic)
- `.jsx` files - React components (frontend UI elements)
- `.css` files - Styling files (colors, layouts, fonts)
- `.json` files - Data storage files
- `.md` files - Documentation (like this file)

### Key Concepts

- **API Endpoint**: A URL that the frontend calls to get/send data to the backend
- **SSE (Server-Sent Events)**: A way for the backend to send data continuously (streaming)
- **State**: Data that a component remembers between interactions
- **localStorage**: Browser storage that persists even after closing the tab
- **Component**: A reusable piece of UI (like a button, a panel, etc.)

---

## Feature #1: Custom System Prompts

### What It Does

Allows users to customize the instructions given to all AI models before they answer questions. For example, you could tell all models to "respond like a professional lawyer" or "explain things simply for a 10-year-old."

### Why It Was Built

Different use cases need different AI behaviors. A coding assistant should behave differently than a creative writing helper. System prompts let users customize this without changing code.

### Files Created (New Files)

**None** - This feature only modified existing files.

### Files Modified

#### 1. `backend/council.py` (Python)

**What this file does**: Contains the core logic for how the council operates.

**What was changed**:
- The `stage1_collect_responses()` function now accepts a `system_prompt` parameter
- When building the message list sent to AI models, if a system prompt is provided, it's added as the first message with role "system"

**Before (simplified)**:
```python
def stage1_collect_responses(user_query):
    messages = [{"role": "user", "content": user_query}]
    # Send to AI models...
```

**After (simplified)**:
```python
def stage1_collect_responses(user_query, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})
    # Send to AI models...
```

**Reasoning**: The AI API expects messages in a specific format. System messages tell the AI how to behave, user messages are the actual question.

#### 2. `backend/main.py` (Python)

**What this file does**: The entry point of the backend server. Defines all API endpoints.

**What was changed**:
- Added `system_prompt` field to the request body schema
- Passes the system prompt through to the council functions

**Code explanation**:
```python
class SendMessageRequest(BaseModel):
    content: str           # The user's question
    system_prompt: Optional[str] = None  # NEW: Optional system prompt
```

**Reasoning**: API endpoints need to know what data they accept. Adding the field here tells the frontend what it can send.

#### 3. `frontend/src/App.jsx` (React/JavaScript)

**What this file does**: The main frontend component that orchestrates everything.

**What was changed**:
- Added `systemPrompt` state variable (stores the current system prompt)
- Added `showSettings` state variable (controls if settings panel is visible)
- Added a collapsible settings panel with a textarea for entering the prompt
- Saves the prompt to localStorage so it persists between browser sessions

**Code explanation**:
```javascript
// State to store the system prompt
const [systemPrompt, setSystemPrompt] = useState(
  localStorage.getItem('systemPrompt') || ''  // Load from storage or empty
);

// When sending a message, include the system prompt
await sendMessageStream(conversationId, content, systemPrompt, ...);
```

**Reasoning**: React uses "state" to track data that can change. localStorage keeps the prompt even after closing the browser.

#### 4. `frontend/src/api.js` (JavaScript)

**What this file does**: Contains all functions that call the backend API.

**What was changed**:
- `sendMessageStream()` function now accepts a `systemPrompt` parameter
- Includes it in the request body sent to the backend

**Code explanation**:
```javascript
export async function sendMessageStream(conversationId, content, systemPrompt, ...) {
  const response = await fetch(`/api/conversations/${conversationId}/message/stream`, {
    method: 'POST',
    body: JSON.stringify({
      content: content,
      system_prompt: systemPrompt,  // NEW: Send system prompt
      // ... other fields
    })
  });
}
```

**Reasoning**: The API function is the bridge between frontend and backend. It packages the data and sends it.

#### 5. `frontend/src/App.css` (CSS)

**What this file does**: Styles for the main application.

**What was changed**:
- Added styles for the settings panel (background color, padding, border)
- Added styles for the system prompt textarea
- Added a blue dot indicator that shows when a system prompt is active

**Code explanation**:
```css
.settings-panel {
  background: #f8fafc;      /* Light gray background */
  padding: 16px;            /* Space inside the panel */
  border-radius: 8px;       /* Rounded corners */
}

.system-prompt-indicator {
  width: 8px;
  height: 8px;
  background: #3b82f6;      /* Blue color */
  border-radius: 50%;       /* Makes it a circle */
}
```

**Reasoning**: CSS makes the UI visually appealing. The blue dot provides visual feedback that a prompt is active.

### How It All Works Together

1. User opens the settings panel by clicking a button
2. User types a system prompt in the textarea
3. React saves it to state and localStorage
4. When user sends a question, the frontend calls `sendMessageStream()` with the prompt
5. Backend receives the prompt and passes it to `stage1_collect_responses()`
6. Each AI model receives: "System: [your prompt]" + "User: [your question]"
7. AI models respond according to the system prompt

---

## Feature #2: Conversation Export

### What It Does

Adds buttons to download conversations as Markdown (.md) or JSON (.json) files for sharing, backup, or documentation purposes.

### Why It Was Built

Users wanted to save and share their council conversations. Markdown is human-readable and works in GitHub, Discord, Notion, etc. JSON is machine-readable for programmatic use.

### Files Created (New Files)

#### 1. `frontend/src/utils/export.js` (JavaScript)

**What this file does**: Contains functions to convert conversation data into downloadable files.

**Full implementation explained**:

```javascript
// Function to export as Markdown
export function exportToMarkdown(conversation) {
  // Start with the title
  let markdown = `# ${conversation.title || 'Conversation'}\n\n`;

  // Add each message
  for (const message of conversation.messages) {
    if (message.role === 'user') {
      // User messages are simple
      markdown += `## User\n${message.content}\n\n`;
    } else {
      // Assistant messages have 3 stages
      markdown += `## Stage 1: Model Responses\n`;
      // Add each model's response...
      markdown += `## Stage 2: Rankings\n`;
      // Add rankings...
      markdown += `## Stage 3: Final Answer\n`;
      // Add final synthesis...
    }
  }

  // Create a downloadable file
  const blob = new Blob([markdown], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);

  // Trigger download
  const link = document.createElement('a');
  link.href = url;
  link.download = `${sanitizeFilename(conversation.title)}.md`;
  link.click();
}
```

**Key concepts explained**:
- `Blob`: A "Binary Large Object" - essentially a file in memory
- `URL.createObjectURL()`: Creates a temporary URL to the blob that browsers can download
- `document.createElement('a')`: Creates an invisible link element
- `link.click()`: Programmatically "clicks" the link to start download

**Reasoning**: This approach works entirely in the browser without server involvement. No data leaves the user's computer.

### Files Modified

#### 1. `frontend/src/components/ChatInterface.jsx` (React)

**What was changed**:
- Added "Export MD" and "Export JSON" buttons to the header
- Buttons only appear when there are messages to export

**Code explanation**:
```jsx
{messages.length > 0 && (
  <div className="export-buttons">
    <button onClick={() => exportToMarkdown(conversation)}>Export MD</button>
    <button onClick={() => exportToJSON(conversation)}>Export JSON</button>
  </div>
)}
```

**Reasoning**: The `&&` operator means "only show this if messages exist." Empty conversations have nothing to export.

#### 2. `frontend/src/components/ChatInterface.css` (CSS)

**What was changed**:
- Added styles for the export buttons (positioning, colors, hover effects)

---

## Feature #3: Question Categories & Tagging

### What It Does

Lets users tag conversations with categories (coding, writing, analysis, etc.) and filter conversations by tag in the sidebar.

### Why It Was Built

As users accumulate conversations, finding specific ones becomes difficult. Tags provide organization and quick filtering.

### Files Created (New Files)

#### 1. `frontend/src/components/TagEditor.jsx` (React)

**What this file does**: A popup editor where users add/remove tags from a conversation.

**Key implementation details**:
```jsx
function TagEditor({ tags, onTagsChange }) {
  const [newTag, setNewTag] = useState('');

  const SUGGESTED_TAGS = ['coding', 'writing', 'analysis', 'research', 'creative'];

  const addTag = (tag) => {
    const normalizedTag = tag.toLowerCase().trim();  // Normalize to lowercase
    if (!tags.includes(normalizedTag)) {
      onTagsChange([...tags, normalizedTag]);
    }
  };

  const removeTag = (tagToRemove) => {
    onTagsChange(tags.filter(tag => tag !== tagToRemove));
  };

  return (
    <div className="tag-editor">
      <input
        value={newTag}
        onChange={(e) => setNewTag(e.target.value)}
        placeholder="Add a tag..."
      />
      {/* Suggested tags as clickable buttons */}
      {SUGGESTED_TAGS.map(tag => (
        <button onClick={() => addTag(tag)}>{tag}</button>
      ))}
      {/* Display current tags with remove buttons */}
      {tags.map(tag => (
        <span className="tag">
          {tag}
          <button onClick={() => removeTag(tag)}>×</button>
        </span>
      ))}
    </div>
  );
}
```

**Reasoning**: Tags are normalized to lowercase to prevent duplicates like "Coding" and "coding".

#### 2. `frontend/src/components/TagEditor.css` (CSS)

**What was changed**: Styles for tag pills, input field, and suggested tag buttons.

### Files Modified

#### 1. `backend/storage.py` (Python)

**What this file does**: Handles saving/loading conversations to/from JSON files.

**What was changed**:
- Added `tags` field to conversation data structure
- Added `update_conversation_tags()` function
- Added `get_all_tags()` function to get all unique tags
- Added `filter_conversations_by_tag()` function

**Code explanation**:
```python
def update_conversation_tags(conversation_id: str, tags: List[str]):
    """Update the tags for a specific conversation."""
    # Load the conversation file
    filepath = f"data/conversations/{conversation_id}.json"
    with open(filepath, 'r') as f:
        conversation = json.load(f)

    # Update tags
    conversation['tags'] = tags

    # Save back to file
    with open(filepath, 'w') as f:
        json.dump(conversation, f, indent=2)
```

**Reasoning**: Tags are stored directly in each conversation's JSON file for simplicity.

#### 2. `backend/main.py` (Python)

**What was changed**:
- Added `PUT /api/conversations/{id}/tags` endpoint to update tags
- Added `GET /api/tags` endpoint to get all unique tags
- Modified `GET /api/conversations` to accept optional `tag` query parameter for filtering

**Code explanation**:
```python
@app.put("/api/conversations/{id}/tags")
async def update_tags(id: str, tags: List[str]):
    return update_conversation_tags(id, tags)

@app.get("/api/tags")
async def get_tags():
    return get_all_tags()  # Returns ['coding', 'writing', 'analysis', ...]
```

#### 3. `frontend/src/components/Sidebar.jsx` (React)

**What was changed**:
- Added tag filter dropdown at the top
- Display tags on each conversation item

#### 4. `frontend/src/api.js` (JavaScript)

**What was changed**:
- Added `updateTags()` function to call the tags endpoint
- Added `getAllTags()` function
- Modified `listConversations()` to accept optional tag filter

---

## Feature #4: Model Configuration UI

### What It Does

A modal panel where users can add/remove AI models from the council, change the chairman model, and configure model settings without editing code.

### Why It Was Built

Previously, changing models required editing Python code. Users wanted to experiment with different model combinations easily.

### Files Created (New Files)

#### 1. `backend/config_api.py` (Python)

**What this file does**: Manages dynamic configuration (which models to use) without code changes.

**Key implementation**:
```python
# Default configuration
DEFAULT_CONFIG = {
    "council_models": [
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3.5-haiku",
        "openai/gpt-4.1-mini",
        "deepseek/deepseek-chat-v3"
    ],
    "chairman_model": "google/gemini-2.5-flash-preview"
}

CONFIG_FILE = "data/council_config.json"

def load_config():
    """Load configuration from file, or return defaults if file doesn't exist."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to file after validation."""
    if len(config.get("council_models", [])) < 2:
        raise ValueError("At least 2 council models required")

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
```

**Reasoning**: Storing config in a JSON file means it persists across server restarts without database setup.

#### 2. `frontend/src/components/ConfigPanel.jsx` (React)

**What this file does**: A modal dialog for configuring models.

**Key features**:
- Autocomplete dropdown with suggested models from OpenRouter
- Add/remove buttons for council models
- Drag handles to reorder models
- Chairman selection dropdown
- "Reset to Defaults" button
- Save/Cancel buttons with validation

**Simplified structure**:
```jsx
function ConfigPanel({ onClose }) {
  const [councilModels, setCouncilModels] = useState([]);
  const [chairmanModel, setChairmanModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);

  // Load current config on mount
  useEffect(() => {
    loadConfig().then(config => {
      setCouncilModels(config.council_models);
      setChairmanModel(config.chairman_model);
    });
    getAvailableModels().then(models => setAvailableModels(models));
  }, []);

  const handleSave = async () => {
    if (councilModels.length < 2) {
      alert("Need at least 2 council models!");
      return;
    }
    await updateConfig({ council_models: councilModels, chairman_model: chairmanModel });
    onClose();
  };

  return (
    <div className="config-modal-overlay">
      <div className="config-panel">
        {/* Council models list with add/remove buttons */}
        {/* Chairman dropdown */}
        {/* Save/Cancel buttons */}
      </div>
    </div>
  );
}
```

#### 3. `frontend/src/components/ConfigPanel.css` (CSS)

**What was changed**: Modal styling, model list items, autocomplete dropdown, buttons.

### Files Modified

#### 1. `backend/main.py` (Python)

**What was changed**:
- Added `GET /api/config` endpoint to retrieve configuration
- Added `PUT /api/config` endpoint to update configuration
- Added `POST /api/config/reset` endpoint to reset to defaults
- Added `GET /api/config/models` endpoint to get list of suggested models

#### 2. `backend/council.py` (Python)

**What was changed**:
- Changed from hardcoded model lists to using `get_council_models()` and `get_chairman_model()` from config_api

**Before**:
```python
COUNCIL_MODELS = ["openai/gpt-4", "anthropic/claude-3", ...]  # Hardcoded
```

**After**:
```python
from .config_api import get_council_models, get_chairman_model
# Uses dynamic config from get_council_models() function
```

---

## Feature #5: Reasoning Model Support

### What It Does

Special handling for AI models like o1, o3, and DeepSeek-R1 that expose their internal "thinking process" (chain-of-thought reasoning).

### Why It Was Built

Newer AI models can show how they think through a problem step-by-step. This is valuable for understanding and validating their reasoning.

### Files Created (New Files)

**None** - Only modified existing files.

### Files Modified

#### 1. `backend/openrouter.py` (Python)

**What was changed**:
- Added detection of `reasoning_details` in API responses
- Passes reasoning data through to the response

**Code explanation**:
```python
async def query_model(model, messages):
    response = await client.post(API_URL, json={...})
    data = response.json()

    result = {
        "content": data["choices"][0]["message"]["content"],
        "usage": data.get("usage", {})
    }

    # NEW: Check for reasoning details
    if "reasoning_details" in data["choices"][0]["message"]:
        result["reasoning_details"] = data["choices"][0]["message"]["reasoning_details"]

    return result
```

**Reasoning**: Different models return reasoning in different formats. We capture whatever the API provides.

#### 2. `frontend/src/components/Stage1.jsx` (React)

**What was changed**:
- Added detection of `reasoning_details` in response data
- Added collapsible "Show Thinking Process" section with amber/gold styling
- Added "Reasoning Model" badge on tabs for models with reasoning

**Code explanation**:
```jsx
{response.reasoning_details && (
  <div className="reasoning-section">
    <button onClick={() => setShowReasoning(!showReasoning)}>
      {showReasoning ? 'Hide' : 'Show'} Thinking Process
    </button>
    {showReasoning && (
      <div className="reasoning-content">
        {formatReasoningDetails(response.reasoning_details)}
      </div>
    )}
  </div>
)}
```

#### 3. `frontend/src/components/Stage1.css` (CSS)

**What was changed**:
- Added amber/gold styling for reasoning sections (#f59e0b color scheme)
- Added asterisk indicator on tabs for reasoning models

---

## Feature #6: Cost Tracking

### What It Does

Shows real-time token usage and cost (in USD) for each query, broken down by stage.

### Why It Was Built

API calls cost money. Users need to see how much each query costs to manage budgets.

### Files Created (New Files)

#### 1. `backend/pricing.py` (Python)

**What this file does**: Contains pricing data for each AI model.

**Implementation**:
```python
# Prices per 1 million tokens (input/output)
MODEL_PRICING = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4.1-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    # ... more models
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """Calculate cost for a model query."""
    pricing = MODEL_PRICING.get(model, {"input": 1.00, "output": 3.00})  # Default pricing

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }
```

**Reasoning**: Pricing is stored locally because OpenRouter doesn't always return cost data. Defaults ensure unknown models still get estimated.

#### 2. `frontend/src/components/CostDisplay.jsx` (React)

**What this file does**: Displays cost information in a user-friendly format.

**Components exported**:
- `CostDisplay`: Full breakdown by stage
- `ModelCostBadge`: Inline cost badge for individual models

#### 3. `frontend/src/components/CostDisplay.css` (CSS)

**What was changed**: Green color scheme for cost displays (#10b981).

### Files Modified

#### 1. `backend/openrouter.py` (Python)

**What was changed**:
- Extract `usage` (prompt_tokens, completion_tokens) from API responses
- Calculate cost using pricing module

#### 2. `backend/council.py` (Python)

**What was changed**:
- Aggregate costs across all models in each stage
- Include cost breakdown in metadata

---

## Feature #7: Confidence Voting

### What It Does

Each AI model reports a confidence score (1-10) with their response. Aggregate confidence is shown in the header.

### Why It Was Built

Confidence scores help users assess response reliability. Low confidence might indicate the question is ambiguous or the model is uncertain.

### Files Created (New Files)

#### 1. `frontend/src/components/ConfidenceDisplay.jsx` (React)

**Components exported**:
- `ConfidenceDisplay`: Full confidence statistics
- `ConfidenceBadge`: Color-coded badge (green=high, red=low)
- `AggregateConfidenceSummary`: Header summary

**Color scheme**:
- 9-10: Green (#059669) - Very high confidence
- 7-8: Teal (#0d9488) - High confidence
- 5-6: Amber (#d97706) - Medium confidence
- 3-4: Orange (#ea580c) - Low confidence
- 1-2: Red (#dc2626) - Very low confidence

### Files Modified

#### 1. `backend/council.py` (Python)

**What was changed**:
- Added `CONFIDENCE_PROMPT_SUFFIX` appended to all queries:
  ```python
  CONFIDENCE_PROMPT_SUFFIX = "\n\nAt the end of your response, on a new line, provide your confidence in this answer using the format: CONFIDENCE: [1-10]"
  ```
- Added `parse_confidence()` function to extract score from response
- Added `extract_response_without_confidence()` to clean display text
- Added `calculate_aggregate_confidence()` to compute statistics

**Parsing logic**:
```python
def parse_confidence(text: str) -> Optional[int]:
    """Extract confidence score from text."""
    patterns = [
        r"CONFIDENCE:\s*(\d+)",
        r"Confidence:\s*(\d+)",
        r"\*\*CONFIDENCE:\*\*\s*(\d+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    return None
```

**Reasoning**: Models format confidence differently. Multiple patterns catch most variations.

---

## Feature #8: Streaming Responses

### What It Does

Shows AI responses as they're being generated, token by token, instead of waiting for the complete response.

### Why It Was Built

Long responses can take 30+ seconds. Streaming provides immediate feedback and better user experience.

### Files Created (New Files)

**None** - Heavily modified existing files.

### Files Modified

#### 1. `backend/openrouter.py` (Python)

**What was changed**:
- Added `query_model_streaming()` function that yields tokens as they arrive
- Added `query_models_parallel_streaming()` to stream multiple models at once

**Implementation (simplified)**:
```python
async def query_model_streaming(model, messages):
    """Stream tokens from a single model."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            'POST',
            API_URL,
            json={"model": model, "messages": messages, "stream": True}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if 'choices' in data:
                        token = data['choices'][0]['delta'].get('content', '')
                        if token:
                            yield {"type": "token", "content": token}
```

**Key concepts**:
- `stream=True`: Tells OpenRouter to stream the response
- `aiter_lines()`: Iterates over response lines as they arrive
- `yield`: Generates values one at a time (generator pattern)

#### 2. `backend/council.py` (Python)

**What was changed**:
- Added `stage1_collect_responses_streaming()` function
- Added `stage3_synthesize_final_streaming()` function

#### 3. `backend/main.py` (Python)

**What was changed**:
- Created `/message/stream` endpoint using Server-Sent Events (SSE)
- Yields events like `stage1_token`, `stage1_complete`, `stage3_token`, etc.

**SSE format**:
```python
async def event_generator():
    async for event in stage1_streaming():
        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
```

#### 4. `frontend/src/App.jsx` (React)

**What was changed**:
- Added `stage1Streaming` state to accumulate partial responses
- Added `stage3Streaming` state for chairman streaming
- Event listener processes incoming SSE events

**Event handling**:
```javascript
const eventSource = new EventSource(url);

eventSource.addEventListener('stage1_token', (e) => {
  const data = JSON.parse(e.data);
  setStage1Streaming(prev => ({
    ...prev,
    [data.model]: (prev[data.model] || '') + data.content
  }));
});
```

#### 5. `frontend/src/components/Stage1.jsx` (React)

**What was changed**:
- Merges completed responses with streaming responses
- Shows pulsing blue dot on tabs still generating
- Shows blinking cursor at end of streaming text

---

## Feature #9: Performance Dashboard

### What It Does

An analytics dashboard showing model win rates, average rankings, costs, and performance over time.

### Why It Was Built

Users wanted to see which models perform best to optimize their council composition.

### Files Created (New Files)

#### 1. `backend/analytics.py` (Python)

**What this file does**: Records query results and calculates statistics.

**Key functions**:
```python
ANALYTICS_FILE = "data/analytics/model_stats.json"

def record_query_result(stage1_results, stage2_results, aggregate_rankings, costs, ...):
    """Record results after each query for analytics."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_performances": [
            {"model": r["model"], "rank": find_rank(r["model"], aggregate_rankings)}
            for r in stage1_results
        ],
        "total_cost": costs["total"],
        # ...
    }

    # Append to analytics file
    data = load_analytics()
    data["records"].append(record)
    save_analytics(data)

def get_model_statistics():
    """Calculate aggregate statistics for all models."""
    data = load_analytics()
    stats = {}

    for record in data["records"]:
        for perf in record["model_performances"]:
            model = perf["model"]
            if model not in stats:
                stats[model] = {"wins": 0, "total": 0, "ranks": []}
            stats[model]["total"] += 1
            stats[model]["ranks"].append(perf["rank"])
            if perf["rank"] == 1:
                stats[model]["wins"] += 1

    # Calculate win rates, average ranks
    for model in stats:
        stats[model]["win_rate"] = stats[model]["wins"] / stats[model]["total"] * 100
        stats[model]["average_rank"] = sum(stats[model]["ranks"]) / len(stats[model]["ranks"])

    return stats
```

#### 2. `frontend/src/components/PerformanceDashboard.jsx` (React)

**What this file does**: Modal dashboard with three tabs.

**Tabs**:
1. **Leaderboard**: Models ranked by win rate, medal icons for top 3
2. **Model Details**: Detailed cards with rank distribution charts
3. **Chairman Stats**: Chairman usage statistics

**Color scheme**: Purple/indigo (#667eea) to distinguish from other panels.

#### 3. `frontend/src/components/PerformanceDashboard.css` (CSS)

### Files Modified

#### 1. `backend/main.py` (Python)

**What was changed**:
- Added `GET /api/analytics` endpoint
- Added `GET /api/analytics/recent` endpoint
- Added `DELETE /api/analytics` endpoint

#### 2. `backend/council.py` (Python)

**What was changed**:
- Calls `record_query_result()` after each query completes

---

## Feature #10: Live Process Monitor

### What It Does

A side panel showing real-time events during council deliberation (stage transitions, model queries, parsing, etc.).

### Why It Was Built

Users wanted visibility into what's happening "behind the scenes" during long queries.

### Files Created (New Files)

#### 1. `backend/process_logger.py` (Python)

**What this file does**: Defines event types and verbosity levels.

**Implementation**:
```python
from enum import IntEnum

class Verbosity(IntEnum):
    SILENT = 0    # No process events
    BASIC = 1     # Stage transitions only
    STANDARD = 2  # Stage + model events
    VERBOSE = 3   # All detailed events

class EventCategory:
    STAGE = "stage"      # Blue
    MODEL = "model"      # Purple
    INFO = "info"        # Gray
    SUCCESS = "success"  # Green
    WARNING = "warning"  # Amber
    ERROR = "error"      # Red
    DATA = "data"        # Teal

def create_process_event(message, category, level, details=None):
    return {
        "message": message,
        "category": category,
        "level": level,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details
    }

# Pre-defined events
def stage_start(stage_name):
    return create_process_event(f"Starting {stage_name}", EventCategory.STAGE, 1)

def model_query_complete(model, duration_ms):
    return create_process_event(f"{model} responded ({duration_ms}ms)", EventCategory.MODEL, 2)
```

#### 2. `frontend/src/components/ProcessMonitor.jsx` (React)

**Features**:
- Verbosity slider (0-3)
- Auto-scrolling event log
- Color-coded events by category
- Collapsible side panel
- Event count badge when collapsed

**Color scheme**: Dark theme (#1a1a2e) to distinguish from main content.

#### 3. `frontend/src/components/ProcessMonitor.css` (CSS)

### Files Modified

#### 1. `backend/main.py` (Python)

**What was changed**:
- Added `verbosity` parameter to streaming endpoint
- Emits `process` SSE events based on verbosity level

---

---

## Feature #11: Chain-of-Thought Orchestration

### What It Does

Forces ALL AI models to provide explicit structured reasoning in three steps: THINKING → ANALYSIS → CONCLUSION. This is different from Feature #5 (Reasoning Model Support) which only captures native reasoning from models that provide it.

### Why It Was Built

While some models (o1, o3) naturally show their thinking, most don't. This feature makes ALL models explain their reasoning process, enabling better evaluation and comparison.

### Files Created (New Files)

#### 1. `frontend/src/components/ReasoningView.jsx` (React)

**What this file does**: Displays the three-step reasoning structure.

**Implementation**:
```jsx
function ReasoningView({ cot, showAll = false }) {
  const [expanded, setExpanded] = useState({
    thinking: false,
    analysis: false,
    conclusion: true  // Conclusion shown by default
  });

  return (
    <div className="reasoning-view">
      {/* Step 1: Thinking (Blue) */}
      <div className="reasoning-step step-1">
        <button onClick={() => setExpanded(e => ({...e, thinking: !e.thinking}))}>
          <span className="step-number">1</span>
          Thinking
        </button>
        {expanded.thinking && <div className="step-content">{cot.thinking}</div>}
      </div>

      {/* Step 2: Analysis (Purple) */}
      <div className="reasoning-step step-2">
        <button onClick={() => setExpanded(e => ({...e, analysis: !e.analysis}))}>
          <span className="step-number">2</span>
          Analysis
        </button>
        {expanded.analysis && <div className="step-content">{cot.analysis}</div>}
      </div>

      {/* Step 3: Conclusion (Green) */}
      <div className="reasoning-step step-3">
        <span className="step-number">3</span>
        Conclusion
        <div className="step-content">{cot.conclusion}</div>
      </div>
    </div>
  );
}
```

**Also exports**:
- `CoTBadge`: Small badge indicating CoT is available
- `CoTToggle`: Toggle control for settings panel

#### 2. `frontend/src/components/ReasoningView.css` (CSS)

**Color scheme**:
- Step 1 (Thinking): Blue (#3b82f6)
- Step 2 (Analysis): Purple (#8b5cf6)
- Step 3 (Conclusion): Green (#22c55e)

### Files Modified

#### 1. `backend/council.py` (Python)

**What was changed**:
- Added `COT_PROMPT_SUFFIX` that gets appended to user queries when CoT mode is enabled:

```python
COT_PROMPT_SUFFIX = """

Please structure your response with the following three sections:

**THINKING:**
Share your initial thoughts, considerations, and how you're approaching this question.

**ANALYSIS:**
Evaluate different perspectives, weigh pros and cons, and work through the reasoning.

**CONCLUSION:**
Provide your final answer based on the above analysis.
"""
```

- Added `parse_cot_response()` function to extract sections:

```python
def parse_cot_response(text: str) -> Optional[dict]:
    """Extract THINKING, ANALYSIS, CONCLUSION sections."""
    result = {}

    # Try multiple formats (models format differently)
    for section in ['thinking', 'analysis', 'conclusion']:
        patterns = [
            rf"\*\*{section.upper()}:\*\*\s*(.*?)(?=\*\*|$)",  # **THINKING:**
            rf"{section.upper()}:\s*(.*?)(?=[A-Z]+:|$)",       # THINKING:
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result[section] = match.group(1).strip()
                break

    return result if len(result) == 3 else None
```

- Modified `stage1_collect_responses()` to accept `use_cot` parameter
- Modified `stage2_collect_rankings()` to evaluate reasoning quality when CoT enabled

**Stage 2 CoT-aware evaluation prompt**:
```python
if use_cot:
    ranking_prompt += """
When evaluating these responses, consider:
1. Reasoning Quality: Is the thinking process logical and thorough?
2. Analysis Depth: Are different perspectives properly explored?
3. Conclusion Accuracy: Does the conclusion follow from the analysis?
4. Overall Coherence: Do the three sections build on each other?
"""
```

**Reasoning**: By evaluating reasoning quality in Stage 2, models that think well get ranked higher, not just models that give superficially good answers.

#### 2. `backend/main.py` (Python)

**What was changed**:
- Added `use_cot` parameter to request body
- Passes parameter through to council functions

#### 3. `frontend/src/App.jsx` (React)

**What was changed**:
- Added `useCot` state (persisted to localStorage)
- Added CoT toggle in settings panel
- Added "CoT" indicator badge in settings bar
- Pass `use_cot` parameter to streaming API

#### 4. `frontend/src/components/Stage1.jsx` (React)

**What was changed**:
- Detects `cot` object in response data
- Renders `ReasoningView` component when CoT data present
- Shows "CoT" badge on tabs with structured reasoning

---

## Feature #12: Multi-Chairman Synthesis

### What It Does

Instead of one chairman creating the final answer, multiple chairman models independently create syntheses. A "supreme chairman" then evaluates all syntheses and selects the best one.

### Why It Was Built

Different AI models have different strengths. One might synthesize better for technical topics, another for creative topics. Using multiple chairmen and selecting the best provides higher quality final answers.

### Files Created (New Files)

#### 1. `backend/multi_chairman.py` (Python)

**What this file does**: Orchestrates multi-chairman synthesis.

**Key functions**:

```python
async def stage3_multi_chairman_synthesis(user_query, stage1_results, stage2_results):
    """Phase A: Multiple chairmen synthesize in parallel."""
    chairman_models = get_multi_chairman_models()  # e.g., [Gemini, Claude, GPT-4]

    # Query all chairmen simultaneously
    tasks = [
        query_model(model, create_synthesis_prompt(user_query, stage1_results, stage2_results))
        for model in chairman_models
    ]
    results = await asyncio.gather(*tasks)

    return [
        {"model": model, "response": result["content"], "cost": result["cost"]}
        for model, result in zip(chairman_models, results)
    ]

async def stage3_supreme_chairman_selection(user_query, syntheses):
    """Phase B: Supreme chairman evaluates and selects best synthesis."""
    supreme_model = get_chairman_model()

    # Create evaluation prompt
    prompt = f"""You are the Supreme Chairman. Below are {len(syntheses)} different syntheses
of the council's responses to: "{user_query}"

{format_syntheses(syntheses)}

Evaluate each synthesis for:
1. Accuracy and completeness
2. Clarity and organization
3. How well it represents the council's collective wisdom

Then select the best one.

Format your response as:
EVALUATION: [Your evaluation of each synthesis]
SELECTED: [Letter A, B, or C]
REASONING: [Why this synthesis is best]
FINAL RESPONSE: [The selected synthesis, optionally improved]
"""

    result = await query_model(supreme_model, prompt)

    # Parse selection
    selected_letter = extract_selected_letter(result["content"])
    selected_model = syntheses[ord(selected_letter) - ord('A')]["model"]

    return {
        "selected_synthesis": selected_letter,
        "selected_model": selected_model,
        "reasoning": extract_reasoning(result["content"]),
        "final_response": extract_final_response(result["content"])
    }
```

**Reasoning**: Parallel synthesis is fast (all chairmen work simultaneously). The supreme chairman adds quality assurance by picking the best result.

#### 2. `frontend/src/components/MultiSynthesis.jsx` (React)

**What this file does**: Displays the multi-chairman synthesis process.

**Components**:
```jsx
function MultiSynthesis({ result, syntheses, isSelecting, selectionStreaming }) {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="multi-synthesis">
      {/* Tabs for each chairman's synthesis */}
      <div className="synthesis-tabs">
        {syntheses.map((synthesis, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}
                       ${result?.selected_synthesis === String.fromCharCode(65 + index) ? 'selected' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {String.fromCharCode(65 + index)}  {/* A, B, C... */}
            {result?.selected_synthesis === String.fromCharCode(65 + index) && '✓'}
          </button>
        ))}
      </div>

      {/* Active synthesis content */}
      <div className="synthesis-content">
        <div className="model-name">{syntheses[activeTab].model}</div>
        <ReactMarkdown>{syntheses[activeTab].response}</ReactMarkdown>
      </div>

      {/* Supreme chairman selection section */}
      {result && (
        <div className="selection-section">
          <h4>Supreme Chairman Selection</h4>
          <details>
            <summary>Show Evaluation Details</summary>
            {result.reasoning}
          </details>
          <div className="final-response">
            <ReactMarkdown>{result.final_response}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
```

#### 3. `frontend/src/components/MultiSynthesis.css` (CSS)

**Color scheme**: Green (#22c55e) to match Stage 3 theme.

### Files Modified

#### 1. `backend/config_api.py` (Python)

**What was changed**:
- Added `multi_chairman_models` to configuration
- Added `get_multi_chairman_models()` getter function

#### 2. `backend/main.py` (Python)

**What was changed**:
- Added `use_multi_chairman` parameter
- Added SSE events: `multi_synthesis_start`, `synthesis_complete`, `multi_synthesis_complete`, `selection_start`, `selection_token`

#### 3. `frontend/src/components/Stage3.jsx` (React)

**What was changed**:
- Detects multi-chairman mode and renders `MultiSynthesis` component
- Shows "Multi-Chairman" badge in header

---

## Feature #13: Weighted Consensus Voting

### What It Does

Models that historically perform well (higher win rates, better average rankings) have their votes weighted more heavily in Stage 2 rankings.

### Why It Was Built

Not all models are equally reliable. Giving more weight to consistently better performers improves consensus quality.

### Files Created (New Files)

#### 1. `backend/weights.py` (Python)

**What this file does**: Calculates model weights based on historical performance.

**Weight calculation formula**:
```python
# Constants
WIN_RATE_FACTOR = 1.5      # Win rate bonus multiplier
RANK_FACTOR = 0.5          # Rank bonus multiplier
CONFIDENCE_FACTOR = 0.1    # Confidence bonus multiplier
MIN_WEIGHT = 0.3           # Minimum weight (floor)
MAX_WEIGHT = 2.5           # Maximum weight (ceiling)
MIN_QUERIES_FOR_WEIGHT = 2 # Minimum queries before historical weighting

def get_model_weights(models: List[str]) -> Dict[str, Any]:
    """Calculate weights for each model based on analytics."""
    stats = get_model_statistics()  # From analytics module
    weights = {}

    for model in models:
        model_stats = stats.get(model)

        if not model_stats or model_stats["total"] < MIN_QUERIES_FOR_WEIGHT:
            # Not enough data, use default weight
            weights[model] = {"weight": 1.0, "has_history": False}
            continue

        # Calculate weight components
        win_rate = model_stats["win_rate"] / 100  # 0.0 to 1.0
        avg_rank = model_stats["average_rank"]     # 1.0 to N
        avg_confidence = model_stats.get("average_confidence", 5)  # 1 to 10

        # Base weight
        weight = 1.0

        # Win rate bonus: models that win more get higher weight
        weight += win_rate * WIN_RATE_FACTOR

        # Rank bonus: models with better (lower) average ranks get bonus
        weight += (1 / avg_rank) * RANK_FACTOR

        # Confidence bonus: models with higher confidence get small bonus
        weight += ((avg_confidence - 1) / 9) * CONFIDENCE_FACTOR

        # Clamp to min/max
        weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weight))

        weights[model] = {
            "weight": weight,
            "has_history": True,
            "win_rate": model_stats["win_rate"],
            "average_rank": avg_rank,
            "total_queries": model_stats["total"]
        }

    # Normalize so average weight is 1.0
    avg_weight = sum(w["weight"] for w in weights.values()) / len(weights)
    for model in weights:
        weights[model]["normalized_weight"] = weights[model]["weight"] / avg_weight

    return weights
```

**Weighted ranking calculation**:
```python
def calculate_weighted_aggregate_rankings(stage2_results, label_to_model, use_weights=True):
    """Calculate aggregate rankings with optional weighting."""
    if not use_weights:
        # Simple average
        return calculate_aggregate_rankings(stage2_results, label_to_model)

    weights = get_model_weights([r["model"] for r in stage2_results])
    model_scores = {}  # model -> list of weighted ranks

    for result in stage2_results:
        voter_model = result["model"]
        voter_weight = weights[voter_model]["normalized_weight"]

        for position, label in enumerate(result["parsed_ranking"], start=1):
            voted_model = label_to_model[label]
            if voted_model not in model_scores:
                model_scores[voted_model] = []
            # Weight the vote: lower rank = better, multiplied by voter weight
            model_scores[voted_model].append(position * voter_weight)

    # Calculate weighted average ranks
    results = []
    for model, scores in model_scores.items():
        results.append({
            "model": model,
            "weighted_average_rank": sum(scores) / len(scores),
            "rankings_count": len(scores),
            "weights_applied": True
        })

    return sorted(results, key=lambda x: x["weighted_average_rank"])
```

**Reasoning**: This creates a "reputation system" where models earn trust through good performance.

### Files Modified

#### 1. `backend/council.py` (Python)

**What was changed**:
- `run_full_council()` accepts `use_weighted_consensus` parameter (default True)
- Uses `calculate_weighted_aggregate_rankings()` when enabled

#### 2. `backend/main.py` (Python)

**What was changed**:
- Added `use_weighted_consensus` parameter
- Added `GET /api/weights` endpoint
- Includes `weights_info` in `stage2_complete` event

#### 3. `frontend/src/components/Stage2.jsx` (React)

**What was changed**:
- Shows "Weighted" badge in aggregate rankings header
- Collapsible weights summary section
- Displays model weights and explains calculation

---

## Feature #14: Early Consensus Exit

### What It Does

When council models strongly agree on the best answer (high confidence, same #1 ranking), skip Stage 3 synthesis entirely and return the winning response directly.

### Why It Was Built

If 4 out of 5 models rank Response A as #1 with high confidence, there's no need to synthesize - just use Response A. This saves time and cost.

### Files Created (New Files)

**None** - Only modified existing files.

### Files Modified

#### 1. `backend/council.py` (Python)

**What was changed**:
- Added consensus detection constants:

```python
CONSENSUS_MAX_AVG_RANK = 1.5      # Top model's avg rank must be ≤ 1.5
CONSENSUS_MIN_AGREEMENT = 0.8     # At least 80% must rank top model #1
CONSENSUS_MIN_CONFIDENCE = 7      # Average confidence must be ≥ 7
```

- Added `detect_consensus()` function:

```python
def detect_consensus(stage1_results, stage2_results, label_to_model, aggregate_rankings, aggregate_confidence):
    """Check if strong consensus exists to skip Stage 3."""

    # Get top-ranked model
    top_model = aggregate_rankings[0]["model"]
    top_avg_rank = aggregate_rankings[0]["weighted_average_rank"]

    # Check criterion 1: Average rank
    if top_avg_rank > CONSENSUS_MAX_AVG_RANK:
        return {"is_consensus": False, "reason": "Average rank too high"}

    # Check criterion 2: First-place agreement
    first_place_votes = 0
    total_voters = len(stage2_results)

    for result in stage2_results:
        if result["parsed_ranking"][0] == model_to_label(top_model):
            first_place_votes += 1

    agreement_ratio = first_place_votes / total_voters
    if agreement_ratio < CONSENSUS_MIN_AGREEMENT:
        return {"is_consensus": False, "reason": "Insufficient agreement"}

    # Check criterion 3: Confidence threshold
    if aggregate_confidence["average"] < CONSENSUS_MIN_CONFIDENCE:
        return {"is_consensus": False, "reason": "Low confidence"}

    # All criteria met - consensus!
    # Find the winning model's Stage 1 response
    consensus_response = next(
        r["response"] for r in stage1_results if r["model"] == top_model
    )

    return {
        "is_consensus": True,
        "consensus_model": top_model,
        "consensus_response": consensus_response,
        "reason": "Strong consensus detected",
        "metrics": {
            "average_rank": top_avg_rank,
            "first_place_votes": first_place_votes,
            "total_voters": total_voters,
            "agreement_ratio": agreement_ratio,
            "average_confidence": aggregate_confidence["average"]
        }
    }
```

- Modified `run_full_council()` to skip Stage 3 when consensus detected

#### 2. `backend/main.py` (Python)

**What was changed**:
- Added `use_early_consensus` parameter
- Added `consensus_detected` SSE event

#### 3. `frontend/src/components/Stage3.jsx` (React)

**What was changed**:
- Special rendering for consensus results (cyan/teal theme)
- Shows metrics grid: winning model, average rank, votes, confidence
- Green notice explaining Stage 3 was skipped

---

## Feature #15: Dynamic Model Routing

### What It Does

Classifies questions into categories (coding, creative, factual, analysis, general) and routes to specialized model pools optimized for each type.

### Why It Was Built

Some models excel at coding, others at creative writing. Routing questions to the best models for each category improves answer quality.

### Files Created (New Files)

#### 1. `backend/router.py` (Python)

**What this file does**: Classifies questions and selects appropriate models.

**Categories and their optimized pools**:
```python
CATEGORIES = {
    "CODING": {
        "description": "Programming, debugging, code review, algorithms",
        "keywords": ["code", "function", "bug", "debug", "programming", "algorithm"],
        "models": ["deepseek/deepseek-chat-v3", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"]
    },
    "CREATIVE": {
        "description": "Writing, brainstorming, storytelling, poetry",
        "keywords": ["write", "story", "creative", "poem", "brainstorm"],
        "models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-2.0-flash"]
    },
    "FACTUAL": {
        "description": "Research, facts, definitions, historical information",
        "keywords": ["what is", "define", "explain", "history", "fact"],
        "models": ["openai/gpt-4o", "google/gemini-2.0-flash", "anthropic/claude-3.5-sonnet"]
    },
    "ANALYSIS": {
        "description": "Comparison, evaluation, logical reasoning, strategy",
        "keywords": ["compare", "analyze", "evaluate", "pros and cons", "strategy"],
        "models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "deepseek/deepseek-chat-v3"]
    },
    "GENERAL": {
        "description": "Broad questions using all configured models",
        "keywords": [],
        "models": None  # Uses all council models
    }
}

# Fast model for classification
CLASSIFIER_MODEL = "google/gemini-2.0-flash-001"

async def classify_question(query: str) -> tuple:
    """Classify a question into a category using LLM."""
    prompt = f"""Classify this question into one category:
CODING, CREATIVE, FACTUAL, ANALYSIS, or GENERAL

Question: "{query}"

Respond with:
CATEGORY: [category]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""

    try:
        result = await query_model(CLASSIFIER_MODEL, prompt)
        # Parse response...
        return (category, confidence, reasoning)
    except:
        # Fallback to keyword classification
        return classify_by_keywords(query)

def classify_by_keywords(query: str) -> tuple:
    """Fallback classification using keywords."""
    query_lower = query.lower()

    for category, info in CATEGORIES.items():
        if any(keyword in query_lower for keyword in info["keywords"]):
            return (category, 0.7, "Matched keywords")

    return ("GENERAL", 0.5, "No specific category detected")

async def route_query(query: str, all_models: List[str]) -> dict:
    """Main routing function."""
    category, confidence, reasoning = await classify_question(query)

    # Get model pool for category
    pool = CATEGORIES[category].get("models")
    if pool is None:
        # GENERAL category uses all models
        selected_models = all_models
        is_routed = False
    else:
        # Intersect with available models
        selected_models = [m for m in pool if m in all_models]
        # Ensure minimum 2 models
        if len(selected_models) < 2:
            selected_models = all_models[:4]
        is_routed = True

    return {
        "category": category,
        "confidence": confidence,
        "reasoning": reasoning,
        "models": selected_models,
        "is_routed": is_routed,
        "original_model_count": len(all_models),
        "routed_model_count": len(selected_models)
    }
```

**Reasoning**: LLM classification is more accurate than keywords alone. Keywords provide fallback when LLM fails.

### Files Modified

#### 1. `backend/council.py` (Python)

**What was changed**:
- `stage1_collect_responses()` accepts optional `models` parameter to override defaults
- `run_full_council()` accepts `use_dynamic_routing` parameter
- Calls `router.route_query()` before Stage 1 when enabled

#### 2. `backend/main.py` (Python)

**What was changed**:
- Added `use_dynamic_routing` parameter
- Added `GET /api/routing/pools` endpoint
- Added `POST /api/routing/classify` endpoint for testing
- Added `routing_start` and `routing_complete` SSE events

#### 3. `frontend/src/components/Stage1.jsx` (React)

**What was changed**:
- Shows routing badge in header with category name
- Color-coded per category: green (coding), purple (creative), blue (factual), amber (analysis), gray (general)
- Hover tooltip shows confidence and reasoning

---

## Feature #16: Confidence-Gated Escalation

### What It Does

Starts with cheaper "Tier 1" models. If confidence is low, escalates to expensive "Tier 2" premium models.

### Why It Was Built

Premium models cost 10-50x more than budget models. Only use them when the cheap models aren't confident enough.

### Files Created (New Files)

#### 1. `backend/escalation.py` (Python)

**What this file does**: Implements tiered model selection.

**Implementation**:
```python
# Escalation thresholds (configured in config_api.py)
DEFAULT_CONFIDENCE_THRESHOLD = 6.0        # Average confidence threshold
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 4      # Any model below this triggers escalation
DEFAULT_AGREEMENT_THRESHOLD = 0.5         # First-place agreement threshold

def should_escalate(aggregate_confidence, stage1_results, stage2_results, label_to_model) -> tuple:
    """Determine if escalation to Tier 2 is needed."""
    thresholds = get_escalation_thresholds()
    reasons = []
    triggers = []

    # Trigger 1: Low average confidence
    if aggregate_confidence["average"] < thresholds["confidence_threshold"]:
        reasons.append(f"Average confidence {aggregate_confidence['average']:.1f} < {thresholds['confidence_threshold']}")
        triggers.append("low_average_confidence")

    # Trigger 2: Any model with very low confidence
    for result in stage1_results:
        if result.get("confidence") and result["confidence"] < thresholds["min_confidence_threshold"]:
            reasons.append(f"{result['model']} confidence {result['confidence']} < {thresholds['min_confidence_threshold']}")
            triggers.append("low_model_confidence")
            break

    # Trigger 3: Low first-place agreement
    top_model = calculate_aggregate_rankings(stage2_results, label_to_model)[0]["model"]
    top_label = model_to_label(top_model, label_to_model)
    first_place_votes = sum(1 for r in stage2_results if r["parsed_ranking"][0] == top_label)
    agreement = first_place_votes / len(stage2_results)

    if agreement < thresholds["agreement_threshold"]:
        reasons.append(f"First-place agreement {agreement:.0%} < {thresholds['agreement_threshold']:.0%}")
        triggers.append("low_agreement")

    should_escalate = len(triggers) > 0
    return (should_escalate, {
        "reasons": reasons,
        "triggers": triggers,
        "metrics": {
            "average_confidence": aggregate_confidence["average"],
            "first_place_agreement": agreement
        }
    })

def merge_tier_results(tier1_results, tier2_results) -> List[dict]:
    """Combine Tier 1 and Tier 2 results."""
    merged = []
    for result in tier1_results:
        merged.append({**result, "tier": 1})
    for result in tier2_results:
        merged.append({**result, "tier": 2})
    return merged
```

#### 2. `frontend/src/components/TierIndicator.jsx` (React)

**Components**:
- `TierIndicator`: Shows "Tier 1 Fast" or "Tier 2 Premium" badge
- `TierBadge`: Small "T1"/"T2" badge for model tabs
- `EscalationBanner`: Shows escalation reasons
- `TierSummary`: Compact summary in header

**Color scheme**:
- Tier 1: Blue (#3b82f6) - fast, cost-effective
- Tier 2: Amber (#f59e0b) - premium, high-quality

### Files Modified

#### 1. `backend/config_api.py` (Python)

**What was changed**:
- Added `tier1_models` and `tier2_models` to configuration
- Added `escalation_confidence_threshold`, `escalation_min_confidence_threshold`, `escalation_agreement_threshold`

#### 2. `backend/main.py` (Python)

**What was changed**:
- Added `use_escalation` parameter
- Implemented full escalation flow: Tier 1 → evaluate → Tier 2 (if needed) → merge

---

## Feature #17: Iterative Refinement Loop

### What It Does

After the chairman creates a synthesis, council models critique it. The chairman revises based on critiques. This repeats until quality converges or max iterations reached.

### Why It Was Built

Even good syntheses can be improved. Iterative critique-and-revise produces higher quality final answers.

### Files Created (New Files)

#### 1. `backend/refinement.py` (Python)

**What this file does**: Implements the critique-revise loop.

**Key functions**:
```python
DEFAULT_MAX_ITERATIONS = 2
DEFAULT_MIN_CRITIQUES_FOR_REVISION = 2

# Phrases indicating non-substantive feedback
NON_SUBSTANTIVE_PHRASES = [
    "no issues", "looks good", "well done", "excellent", "no changes needed",
    "comprehensive", "thorough", "well-structured", "accurate"
]

def is_substantive_critique(critique: str) -> bool:
    """Check if critique contains actionable feedback."""
    critique_lower = critique.lower()

    # If it contains mostly praise phrases, it's not substantive
    for phrase in NON_SUBSTANTIVE_PHRASES:
        if phrase in critique_lower and len(critique) < 200:
            return False

    return True

async def collect_critiques(question: str, draft: str, models: List[str]) -> tuple:
    """Council models critique the current draft."""
    prompt = f"""Review this synthesis and provide constructive criticism:

ORIGINAL QUESTION: {question}

DRAFT SYNTHESIS:
{draft}

Identify any issues with:
- Accuracy or factual errors
- Missing important information
- Unclear or confusing explanations
- Logical inconsistencies

If the synthesis is excellent, simply say "No significant issues found."
"""

    tasks = [query_model(model, prompt) for model in models]
    results = await asyncio.gather(*tasks)

    critiques = [
        {
            "model": model,
            "critique": result["content"],
            "is_substantive": is_substantive_critique(result["content"])
        }
        for model, result in zip(models, results)
    ]

    return (critiques, aggregate_costs(results))

async def generate_revision(question: str, draft: str, critiques: List[dict], chairman: str) -> tuple:
    """Chairman revises based on critiques."""
    substantive_critiques = [c for c in critiques if c["is_substantive"]]

    prompt = f"""Revise this synthesis based on the critiques:

ORIGINAL QUESTION: {question}

CURRENT DRAFT:
{draft}

CRITIQUES:
{format_critiques(substantive_critiques)}

Address each valid criticism while maintaining the strengths of the original.
"""

    result = await query_model(chairman, prompt)
    return (result["content"], result["cost"])

async def run_refinement_loop_streaming(question, initial_draft, max_iterations=DEFAULT_MAX_ITERATIONS, ...):
    """Full refinement loop with streaming."""
    current_draft = initial_draft
    iterations = []

    for i in range(max_iterations):
        yield {"type": "iteration_start", "iteration": i + 1}

        # Collect critiques
        yield {"type": "critiques_start", "iteration": i + 1}
        critiques, critique_cost = await collect_critiques(question, current_draft, council_models)

        for critique in critiques:
            yield {"type": "critique_complete", **critique}

        substantive_count = count_substantive_critiques(critiques)
        yield {"type": "critiques_complete", "substantive_count": substantive_count}

        # Check for convergence
        if substantive_count < DEFAULT_MIN_CRITIQUES_FOR_REVISION:
            yield {"type": "refinement_converged", "iteration": i + 1, "reason": "Quality converged"}
            break

        # Generate revision
        yield {"type": "revision_start", "iteration": i + 1}
        async for token_event in generate_revision_streaming(...):
            yield token_event

        iterations.append({"critiques": critiques, "revision": current_draft})

    yield {"type": "refinement_complete", "iterations": iterations, "final_response": current_draft}
```

**Reasoning**: Convergence detection prevents unnecessary iterations when the synthesis is already good.

#### 2. `frontend/src/components/RefinementView.jsx` (React)

**What this file does**: Displays refinement iterations.

**Components**:
- `RefinementView`: Shows iteration history with critiques and revisions
- `RefinementBadge`: Shows iteration count in Stage 3 header
- `RefinementToggle`: Toggle with max iterations dropdown

**Color scheme**: Purple/violet (#7c3aed, #8b5cf6)

### Files Modified

#### 1. `backend/main.py` (Python)

**What was changed**:
- Added `use_refinement` and `refinement_max_iterations` parameters
- Refinement runs after Stage 3 completes (if enabled)

#### 2. `frontend/src/components/Stage3.jsx` (React)

**What was changed**:
- Renders `RefinementView` below final response

---

## Feature #18: Adversarial Validation

### What It Does

A "devil's advocate" model critically reviews the final synthesis, looking for errors, biases, and weaknesses. If serious issues found, chairman revises.

### Why It Was Built

Even after refinement, syntheses might have blind spots. An adversarial review catches errors the council missed.

### Files Created (New Files)

#### 1. `backend/adversary.py` (Python)

**What this file does**: Implements adversarial validation.

**Implementation**:
```python
DEFAULT_ADVERSARY_MODEL = "google/gemini-2.0-flash-001"

# Phrases indicating no real issues
NO_ISSUES_PHRASES = [
    "no issues", "well-structured", "accurate", "comprehensive",
    "no significant problems", "solid response"
]

# Severities that trigger revision
REVISION_THRESHOLD = ["critical", "major"]

def has_genuine_issues(critique: str) -> bool:
    """Check if critique identifies real issues."""
    critique_lower = critique.lower()
    for phrase in NO_ISSUES_PHRASES:
        if phrase in critique_lower:
            return False
    return True

def parse_severity(critique: str) -> str:
    """Extract severity level from critique."""
    # Look for "SEVERITY: X" pattern
    match = re.search(r"SEVERITY:\s*(critical|major|minor|none)", critique, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "none"

async def run_adversarial_validation_streaming(question, synthesis, adversary_model, chairman_model):
    """Full adversarial validation with streaming."""

    # Phase 1: Adversary critiques
    yield {"type": "adversary_start", "adversary_model": adversary_model}

    adversary_prompt = f"""You are a Devil's Advocate. Critically review this synthesis:

QUESTION: {question}

SYNTHESIS:
{synthesis}

Look for:
- Factual errors or inaccuracies
- Logical fallacies or inconsistencies
- Missing important perspectives
- Potential biases
- Unclear or misleading statements

Format:
ISSUES: [List specific issues found, or "None identified"]
SEVERITY: [critical/major/minor/none]
RECOMMENDATIONS: [Specific improvements if severity is critical/major]
"""

    async for token_event in query_model_streaming(adversary_model, adversary_prompt):
        yield {"type": "adversary_token", "content": token_event["content"]}

    critique = final_adversary_response
    has_issues = has_genuine_issues(critique)
    severity = parse_severity(critique)

    yield {"type": "adversary_complete", "critique": critique, "has_issues": has_issues, "severity": severity}

    # Phase 2: Revision if needed
    if severity in REVISION_THRESHOLD:
        yield {"type": "adversary_revision_start"}

        revision_prompt = f"""Revise this synthesis based on the adversary's critique:

ORIGINAL: {synthesis}

CRITIQUE: {critique}

Address the identified issues while preserving what was done well.
"""

        async for token_event in query_model_streaming(chairman_model, revision_prompt):
            yield {"type": "adversary_revision_token", "content": token_event["content"]}

        yield {"type": "adversary_revision_complete", "revision": final_revision}
        revised = True
    else:
        revised = False

    yield {
        "type": "adversary_validation_complete",
        "critique": critique,
        "has_issues": has_issues,
        "severity": severity,
        "revised": revised,
        "revision": final_revision if revised else None
    }
```

#### 2. `frontend/src/components/AdversaryReview.jsx` (React)

**Components**:
- `AdversaryReview`: Shows critique and optional revision
- `AdversaryBadge`: Shows severity/result in Stage 3 header
- `AdversaryToggle`: Toggle control

**Color scheme**: Red/rose (#e11d48, #f87171)

**Result indicators**:
- "Passed" (green): No issues found
- "Revised" (blue): Issues found and fixed
- "Issues Noted" (amber): Minor issues, no revision

---

## Feature #19: Debate Mode

### What It Does

Instead of normal council process, models engage in a structured debate:
1. Round 1: Each model states their position
2. Round 2: Each model critiques another's position
3. Round 3: Each model defends their position (optional)
4. Final: Chairman evaluates debate and synthesizes

### Why It Was Built

Debates force models to defend their reasoning and identify weaknesses in others' arguments. This produces more thoroughly considered answers.

### Files Created (New Files)

#### 1. `backend/debate.py` (Python)

**What this file does**: Orchestrates multi-round debates.

**Implementation**:
```python
DEFAULT_NUM_ROUNDS = 3
DEFAULT_INCLUDE_REBUTTAL = True

def get_critique_pairs(num_models: int) -> List[tuple]:
    """Create rotation pairing for critiques.
    Model 0 critiques Model 1
    Model 1 critiques Model 2
    ...
    Model N-1 critiques Model 0
    """
    return [(i, (i + 1) % num_models) for i in range(num_models)]

async def run_debate_streaming(question, models, include_rebuttal=True, chairman_model=None):
    """Full streaming debate orchestration."""

    # Assign anonymous labels (Position A, Position B, etc.)
    model_to_label = {model: f"Position {chr(65 + i)}" for i, model in enumerate(models)}
    label_to_model = {v: k for k, v in model_to_label.items()}

    yield {
        "type": "debate_start",
        "models": models,
        "num_rounds": 3 if include_rebuttal else 2,
        "model_to_label": model_to_label,
        "label_to_model": label_to_model
    }

    # Round 1: Initial Positions
    yield {"type": "round1_start"}
    positions = {}
    position_prompt = f"""State your position on this question: {question}

Provide a clear, well-reasoned answer. Be specific and support your points."""

    for model in models:
        result = await query_model(model, position_prompt)
        positions[model] = result["content"]
        yield {
            "type": "position_complete",
            "label": model_to_label[model],
            "model": model,
            "position": result["content"]
        }

    yield {"type": "round1_complete", "positions": list(positions.values())}

    # Round 2: Critiques
    yield {"type": "round2_start"}
    critiques = []
    pairs = get_critique_pairs(len(models))

    for critic_idx, target_idx in pairs:
        critic_model = models[critic_idx]
        target_model = models[target_idx]
        target_position = positions[target_model]

        critique_prompt = f"""Question: {question}

{model_to_label[target_model]} argues:
{target_position}

Critically evaluate this position. Identify weaknesses, logical flaws, or missing considerations."""

        result = await query_model(critic_model, critique_prompt)
        critiques.append({
            "critic_label": model_to_label[critic_model],
            "target_label": model_to_label[target_model],
            "critique": result["content"]
        })
        yield {"type": "debate_critique_complete", **critiques[-1]}

    yield {"type": "round2_complete", "critiques": critiques}

    # Round 3: Rebuttals (optional)
    if include_rebuttal:
        yield {"type": "round3_start"}
        rebuttals = []

        for model in models:
            # Find critique of this model
            my_label = model_to_label[model]
            critique_of_me = next(c for c in critiques if c["target_label"] == my_label)

            rebuttal_prompt = f"""Your position was criticized:

{critique_of_me["critique"]}

Defend your position. Address the criticisms while acknowledging valid points."""

            result = await query_model(model, rebuttal_prompt)
            rebuttals.append({
                "label": my_label,
                "model": model,
                "rebuttal": result["content"]
            })
            yield {"type": "rebuttal_complete", **rebuttals[-1]}

        yield {"type": "round3_complete", "rebuttals": rebuttals}

    # Final: Chairman Judgment
    yield {"type": "judgment_start", "model": chairman_model}

    transcript = format_debate_transcript(question, positions, critiques, rebuttals if include_rebuttal else None)

    judgment_prompt = f"""As the judge, evaluate this debate:

{transcript}

Consider:
1. Strength of initial arguments
2. Quality of critiques
3. Effectiveness of rebuttals
4. Overall reasoning quality

Determine the winner and synthesize the best answer incorporating insights from all positions."""

    async for token_event in query_model_streaming(chairman_model, judgment_prompt):
        yield {"type": "judgment_token", "content": token_event["content"]}

    yield {"type": "judgment_complete", "judgment": final_judgment}
    yield {"type": "debate_complete", "positions": positions, "critiques": critiques, ...}
```

#### 2. `frontend/src/components/DebateView.jsx` (React)

**What this file does**: Visualizes the debate flow.

**Sections**:
1. Round 1: Position cards for each model
2. Round 2: Critique flows showing critic → target relationships
3. Round 3: Rebuttal cards (if enabled)
4. Final: Chairman judgment with streaming

**Color scheme**: Orange/amber (#fbbf24, #d97706)

### Files Modified

- `backend/main.py`: Added `use_debate` and `include_rebuttal` parameters
- `frontend/src/components/Stage3.jsx`: Renders `DebateView` when debate mode active

---

## Feature #20: Sub-Question Decomposition

### What It Does

For complex multi-part questions, breaks them into sub-questions, runs separate "mini-councils" for each, then merges the answers.

### Why It Was Built

Complex questions like "Compare X and Y, then recommend the best approach for scenario Z" have multiple parts. Answering each part separately produces better results.

### Files Created (New Files)

#### 1. `backend/decompose.py` (Python)

**What this file does**: Detects complex questions and orchestrates decomposition.

**Implementation**:
```python
DECOMPOSER_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_COMPLEXITY_THRESHOLD = 0.6
DEFAULT_MAX_SUB_QUESTIONS = 5

# Keywords indicating multi-part questions
COMPLEXITY_KEYWORDS = [
    "compare and contrast", "step by step", "first...then", "analyze...and recommend",
    "multiple", "several", "list", "pros and cons", "on one hand...on the other"
]

async def assess_complexity(query: str) -> dict:
    """Determine if question is complex enough to decompose."""

    # Keyword heuristic
    keyword_score = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in query.lower())
    keyword_complex = keyword_score >= 2

    # LLM classification
    prompt = f"""Is this question complex enough to benefit from being broken into sub-questions?

Question: "{query}"

A complex question has multiple distinct parts that could be answered separately.

Respond with:
IS_COMPLEX: [yes/no]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""

    result = await query_model(DECOMPOSER_MODEL, prompt)
    llm_complex = parse_is_complex(result["content"])
    llm_confidence = parse_confidence(result["content"])

    # Combine heuristic and LLM
    is_complex = llm_complex or keyword_complex
    confidence = max(llm_confidence, 0.7 if keyword_complex else 0.3)

    return {
        "is_complex": is_complex and confidence >= DEFAULT_COMPLEXITY_THRESHOLD,
        "confidence": confidence,
        "reasoning": result["content"]
    }

async def generate_sub_questions(query: str, chairman_model: str) -> List[str]:
    """Chairman generates focused sub-questions."""
    prompt = f"""Break this complex question into 2-5 focused sub-questions:

Question: "{query}"

Each sub-question should:
- Address one specific aspect
- Be answerable independently
- Together cover the full original question

Format:
1. [sub-question 1]
2. [sub-question 2]
...
"""

    result = await query_model(chairman_model, prompt)
    return parse_numbered_list(result["content"])

async def run_mini_council(sub_question: str, models: List[str]) -> dict:
    """Run all models on a single sub-question."""
    tasks = [query_model(model, sub_question) for model in models]
    results = await asyncio.gather(*tasks)

    # Find best answer (highest confidence)
    best_result = max(
        zip(models, results),
        key=lambda x: parse_confidence(x[1]["content"]) or 5
    )

    return {
        "sub_question": sub_question,
        "best_model": best_result[0],
        "best_answer": extract_answer(best_result[1]["content"]),
        "all_responses": list(zip(models, results))
    }

async def merge_sub_answers(question: str, sub_results: List[dict], chairman_model: str):
    """Chairman synthesizes all sub-answers into final response."""
    prompt = f"""Synthesize these sub-answers into a complete response:

ORIGINAL QUESTION: {question}

SUB-ANSWERS:
{format_sub_results(sub_results)}

Create a unified, coherent response that integrates all the sub-answers.
"""

    return await query_model(chairman_model, prompt)

async def run_decomposition_streaming(question, council_models, chairman_model):
    """Full decomposition flow with streaming."""

    yield {"type": "decomposition_start"}

    # Assess complexity
    complexity = await assess_complexity(question)
    yield {"type": "complexity_analyzed", **complexity}

    if not complexity["is_complex"]:
        yield {"type": "decomposition_skip", "reason": "Question not complex enough"}
        return  # Fall through to normal council

    # Generate sub-questions
    sub_questions = await generate_sub_questions(question, chairman_model)
    yield {"type": "sub_questions_generated", "sub_questions": sub_questions}

    # Run mini-councils for each
    sub_results = []
    for i, sq in enumerate(sub_questions):
        yield {"type": "sub_council_start", "index": i}

        result = await run_mini_council(sq, council_models)
        sub_results.append(result)

        yield {"type": "sub_council_complete", "index": i, **result}

    yield {"type": "all_sub_councils_complete", "results": sub_results}

    # Merge answers
    yield {"type": "merge_start", "chairman_model": chairman_model}
    async for token in query_model_streaming(chairman_model, merge_prompt):
        yield {"type": "merge_token", "content": token}

    yield {"type": "decomposition_complete", ...}
```

#### 2. `frontend/src/components/DecomposedView.jsx` (React)

**Components**:
- `DecomposedView`: Shows sub-questions, sub-answers, and merge
- `DecompositionBadge`: Shows sub-question count
- `DecompositionToggle`: Toggle control

**Color scheme**: Teal/cyan (#22d3d1, #14b8a6)

---

## Feature #21: Response Caching Layer (Semantic Cache)

### What It Does

Stores question-answer pairs with vector embeddings. When a similar question is asked, returns the cached answer instead of re-querying all models.

### Why It Was Built

API calls are expensive. If someone asks "What is Python?" and later asks "Explain Python", these are semantically similar - the cached answer can be reused.

### Files Created (New Files)

#### 1. `backend/embeddings.py` (Python)

**What this file does**: Generates text embeddings for semantic similarity.

**Implementation**:
```python
import httpx
import hashlib
import math
from typing import List, Optional

# OpenRouter endpoint for embeddings
EMBEDDINGS_API_URL = "https://openrouter.ai/api/v1/embeddings"

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Dimensions for this model
HASH_EMBEDDING_DIMENSION = 256  # Fallback dimension

async def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> Optional[List[float]]:
    """Get embedding vector from OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {"model": model, "input": text}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(EMBEDDINGS_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                return data['data'][0].get('embedding')
            return None
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_hash_embedding(text: str, dimension: int = HASH_EMBEDDING_DIMENSION) -> List[float]:
    """Fallback: Generate deterministic hash-based embedding.

    Not as semantically meaningful as real embeddings, but works
    for exact and near-exact matches when API is unavailable.
    """
    normalized = text.lower().strip()
    embedding = []

    for i in range(dimension):
        # Create unique seed for each dimension
        seed = f"{i}:{normalized}"
        hash_bytes = hashlib.sha256(seed.encode()).digest()

        # Convert to float between -1 and 1
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big', signed=True)
        normalized_value = hash_int / (2**63)
        embedding.append(normalized_value)

    # Normalize to unit length
    magnitude = math.sqrt(sum(x*x for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding

async def get_embedding_with_fallback(text: str, use_api: bool = True) -> dict:
    """Get embedding with fallback to hash-based approach."""
    if use_api:
        embedding = await get_embedding(text)
        if embedding:
            return {
                'embedding': embedding,
                'method': 'api',
                'model': DEFAULT_EMBEDDING_MODEL,
                'dimension': len(embedding)
            }

    # Fallback to hash
    embedding = get_hash_embedding(text)
    return {
        'embedding': embedding,
        'method': 'hash',
        'model': 'hash',
        'dimension': len(embedding)
    }

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Returns value between -1 and 1:
    - 1 means identical
    - 0 means orthogonal (unrelated)
    - -1 means opposite
    """
    if len(vec1) != len(vec2):
        return 0.0  # Incompatible dimensions

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(x * x for x in vec1))
    mag2 = math.sqrt(sum(x * x for x in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)
```

**Key concepts**:
- **Embedding**: A list of numbers (vector) representing the "meaning" of text
- **Cosine Similarity**: Measures how similar two vectors are (angle between them)
- **Fallback**: Hash-based embeddings work for exact matches when API unavailable

#### 2. `backend/cache.py` (Python)

**What this file does**: Manages the semantic cache storage and retrieval.

**Implementation**:
```python
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .embeddings import get_embedding_with_fallback, cosine_similarity

# Storage locations
CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "semantic_cache.json")
CACHE_STATS_FILE = os.path.join(CACHE_DIR, "cache_stats.json")

# Configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.92  # 92% similarity required for cache hit
DEFAULT_MAX_CACHE_ENTRIES = 1000
DEFAULT_USE_API_EMBEDDINGS = True

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def load_cache() -> Dict[str, Any]:
    """Load cache from JSON file."""
    ensure_cache_dir()

    if not os.path.exists(CACHE_FILE):
        return {"entries": [], "last_updated": None}

    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"entries": [], "last_updated": None}

def save_cache(data: Dict[str, Any]):
    """Save cache to JSON file."""
    ensure_cache_dir()
    data["last_updated"] = datetime.utcnow().isoformat()

    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

async def add_cache_entry(
    query: str,
    response: Dict[str, Any],
    system_prompt: Optional[str] = None,
    max_entries: int = DEFAULT_MAX_CACHE_ENTRIES
) -> Dict[str, Any]:
    """Add a new query-response pair to the cache."""

    # Generate embedding for the query
    embedding_result = await get_embedding_with_fallback(query)

    # Create cache entry
    entry = {
        "id": datetime.utcnow().isoformat() + "-" + str(hash(query) % 10000),
        "query": query,
        "system_prompt": system_prompt,  # Include for cache key isolation
        "embedding": embedding_result['embedding'],
        "embedding_method": embedding_result['method'],
        "response": response,  # Full council response (stage1, stage2, stage3)
        "created_at": datetime.utcnow().isoformat(),
        "hit_count": 0,
        "last_hit": None
    }

    # Load and update cache
    cache = load_cache()
    cache["entries"].append(entry)

    # Enforce max entries (evict oldest)
    if len(cache["entries"]) > max_entries:
        cache["entries"].sort(key=lambda x: x.get("created_at", ""), reverse=True)
        cache["entries"] = cache["entries"][:max_entries]

    save_cache(cache)

    return {
        "id": entry["id"],
        "query": query,
        "embedding_method": embedding_result['method'],
        "cache_size": len(cache["entries"])
    }

async def search_cache(
    query: str,
    system_prompt: Optional[str] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """Search cache for a similar query."""

    cache = load_cache()
    if not cache.get("entries"):
        return None

    # Generate embedding for the query
    embedding_result = await get_embedding_with_fallback(query)
    query_embedding = embedding_result['embedding']

    # Filter by system_prompt (for isolation)
    filtered_entries = [
        e for e in cache["entries"]
        if e.get("system_prompt") == system_prompt
    ]

    if not filtered_entries:
        return None

    # Find best match
    best_match = None
    best_score = similarity_threshold  # Start at threshold

    for entry in filtered_entries:
        stored_embedding = entry.get("embedding", [])

        # Skip if dimensions don't match
        if len(stored_embedding) != len(query_embedding):
            continue

        similarity = cosine_similarity(query_embedding, stored_embedding)

        if similarity > best_score:
            best_score = similarity
            best_match = entry

    if best_match:
        # Update hit statistics
        cache = load_cache()
        for entry in cache["entries"]:
            if entry.get("id") == best_match.get("id"):
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                entry["last_hit"] = datetime.utcnow().isoformat()
                break
        save_cache(cache)

        return {
            "response": best_match.get("response"),
            "similarity": best_score,
            "cached_query": best_match.get("query"),
            "cache_id": best_match.get("id")
        }

    return None

def record_cache_hit(cost_saved: float = 0.0, time_saved_ms: int = 0):
    """Record a cache hit in statistics."""
    stats = load_cache_stats()
    stats["total_queries"] = stats.get("total_queries", 0) + 1
    stats["cache_hits"] = stats.get("cache_hits", 0) + 1
    stats["total_cost_saved"] = stats.get("total_cost_saved", 0.0) + cost_saved
    stats["total_time_saved_ms"] = stats.get("total_time_saved_ms", 0) + time_saved_ms
    save_cache_stats(stats)

def record_cache_miss():
    """Record a cache miss in statistics."""
    stats = load_cache_stats()
    stats["total_queries"] = stats.get("total_queries", 0) + 1
    stats["cache_misses"] = stats.get("cache_misses", 0) + 1
    save_cache_stats(stats)

def get_cache_info() -> Dict[str, Any]:
    """Get cache information for display."""
    cache = load_cache()
    stats = load_cache_stats()

    return {
        "cache_size": len(cache.get("entries", [])),
        "max_entries": DEFAULT_MAX_CACHE_ENTRIES,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "stats": stats
    }

def clear_cache() -> Dict[str, Any]:
    """Clear all cache entries."""
    cache = load_cache()
    entries_cleared = len(cache.get("entries", []))
    save_cache({"entries": [], "last_updated": None})
    return {"success": True, "entries_cleared": entries_cleared}
```

**Key design decisions**:
1. **System prompt isolation**: Queries with different system prompts are cached separately
2. **92% threshold**: High threshold ensures only semantically equivalent queries match
3. **LRU eviction**: When cache is full, oldest entries are removed
4. **Statistics tracking**: Track hits, misses, cost/time saved

### Files Modified

#### 1. `backend/main.py` (Python)

**What was changed**:
- Added `use_cache` parameter to request body
- Added `cache_similarity_threshold` parameter
- Cache check runs before normal council flow
- Added 8 cache endpoints:
  - `GET /api/cache/config` - Get cache configuration
  - `GET /api/cache/info` - Get cache info and stats
  - `GET /api/cache/stats` - Get hit/miss statistics
  - `GET /api/cache/entries` - List cached entries (paginated)
  - `POST /api/cache/search` - Test cache search
  - `DELETE /api/cache` - Clear all cache
  - `DELETE /api/cache/stats` - Clear statistics
  - `DELETE /api/cache/{id}` - Delete specific entry
  - `GET /api/embeddings/config` - Get embedding configuration
- Added SSE events: `cache_check_start`, `cache_hit`, `cache_miss`, `cache_stored`

**Cache flow in event_generator**:
```python
# Early in the flow, before normal council
if use_cache:
    yield create_event("cache_check_start", {"query": content})

    cache_result = await search_cache(
        content,
        system_prompt=system_prompt,
        similarity_threshold=cache_similarity_threshold
    )

    if cache_result:
        # Cache hit - return cached response
        record_cache_hit(cost_saved=estimated_cost, time_saved_ms=estimated_time)
        yield create_event("cache_hit", {
            "similarity": cache_result["similarity"],
            "cached_query": cache_result["cached_query"]
        })
        yield create_event("complete", cache_result["response"])
        return

    # Cache miss - continue to normal flow
    record_cache_miss()
    yield create_event("cache_miss", {})

# ... normal council flow ...

# After completion, store in cache
if use_cache:
    cache_entry = await add_cache_entry(content, full_response, system_prompt)
    yield create_event("cache_stored", cache_entry)
```

#### 2. `frontend/src/api.js` (JavaScript)

**What was changed**:
- Added `useCache` parameter to `sendMessageStream()`
- Added 9 cache API functions:
```javascript
export async function getCacheConfig() { ... }
export async function getCacheInfo() { ... }
export async function getCacheStats() { ... }
export async function getCacheEntries(limit = 50, offset = 0) { ... }
export async function clearCache() { ... }
export async function clearCacheStats() { ... }
export async function deleteCacheEntry(cacheId) { ... }
export async function searchCache(query, systemPrompt = null) { ... }
export async function getEmbeddingsConfig() { ... }
```

#### 3. `frontend/src/App.jsx` (React)

**What was changed**:
- Added `useCache` state (persisted to localStorage)
- Added `handleCacheChange()` handler
- Added SSE event handlers for cache events
- Added "CA" indicator badge in settings bar
- Added cache toggle in settings panel

#### 4. `frontend/src/App.css` (CSS)

**What was changed**:
- Added green theme styling for cache toggle (#10b981, #34d399)
- Added styles for `.cache-section`, `.cache-toggle-wrapper`, `.settings-cache-indicator`

---

## Summary

This document has covered all 21 features implemented in LLM Council:

| Step | Feature | Complexity | New Files | Theme Color |
|------|---------|------------|-----------|-------------|
| 1 | Custom System Prompts | Easy | 0 | Blue |
| 2 | Conversation Export | Easy | 1 | - |
| 3 | Question Categories & Tagging | Medium | 2 | - |
| 4 | Model Configuration UI | Medium | 2 | Blue |
| 5 | Reasoning Model Support | Medium | 0 | Amber |
| 6 | Cost Tracking | Medium | 2 | Green |
| 7 | Confidence Voting | Medium | 2 | Multi-color |
| 8 | Streaming Responses | Hard | 0 | Blue |
| 9 | Performance Dashboard | Medium | 2 | Purple |
| 10 | Live Process Monitor | Medium-Hard | 2 | Dark |
| 11 | Chain-of-Thought | Medium | 2 | Blue/Purple/Green |
| 12 | Multi-Chairman Synthesis | Medium | 2 | Green |
| 13 | Weighted Consensus | Easy | 1 | Amber |
| 14 | Early Consensus Exit | Easy | 0 | Cyan |
| 15 | Dynamic Model Routing | Hard | 1 | Orange |
| 16 | Confidence-Gated Escalation | Medium | 2 | Pink/Amber |
| 17 | Iterative Refinement | Medium | 2 | Purple |
| 18 | Adversarial Validation | Medium | 2 | Red |
| 19 | Debate Mode | Hard | 2 | Orange |
| 20 | Sub-Question Decomposition | Hard | 2 | Teal |
| 21 | Response Caching | Hard | 2 | Green |

### Key Patterns Used Throughout

1. **Backend Module Pattern**: Each feature has a dedicated Python module with clear functions
2. **Config-Driven**: Settings stored in JSON files, not hardcoded
3. **SSE Events**: Real-time updates via Server-Sent Events
4. **Component Pattern**: React components export main view + badge + toggle
5. **localStorage Persistence**: User preferences saved in browser
6. **Graceful Degradation**: Features continue working even when parts fail
7. **Color Theming**: Each feature has a distinct color for visual clarity

### Running the Project

1. **Backend**: `cd backend && python -m backend.main`
2. **Frontend**: `cd frontend && npm run dev`
3. **Access**: Open http://localhost:5173 in your browser

### Continuing Development

To add new features:
1. Study similar existing features as templates
2. Create backend module in `backend/`
3. Add API endpoints in `backend/main.py`
4. Create frontend components in `frontend/src/components/`
5. Update `frontend/src/App.jsx` with state and handlers
6. Update `frontend/src/api.js` with API functions
7. Document in `CLAUDE.md` and this guide
