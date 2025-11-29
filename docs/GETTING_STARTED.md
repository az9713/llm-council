# Getting Started Guide

This guide will walk you through setting up LLM Council from scratch. No prior experience with LLM applications is required.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding What You Need](#understanding-what-you-need)
3. [Step-by-Step Installation](#step-by-step-installation)
4. [Running the Application](#running-the-application)
5. [Your First Query](#your-first-query)
6. [Customizing Models](#customizing-models)
7. [Understanding Costs](#understanding-costs)

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

### 1. Python 3.10 or higher

Check if Python is installed:
```bash
python --version
# Should show Python 3.10.x or higher
```

If not installed, download from [python.org](https://www.python.org/downloads/).

### 2. Node.js 18 or higher

Check if Node.js is installed:
```bash
node --version
# Should show v18.x.x or higher
```

If not installed, download from [nodejs.org](https://nodejs.org/).

### 3. uv (Python package manager)

This project uses `uv` for Python dependency management (faster than pip).

Install uv:
```bash
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip:
pip install uv
```

Verify installation:
```bash
uv --version
```

### 4. OpenRouter API Key

OpenRouter is a service that provides unified access to multiple LLM providers (OpenAI, Google, Anthropic, etc.) through a single API.

**To get an API key:**

1. Go to [openrouter.ai](https://openrouter.ai/)
2. Click "Sign Up" and create an account
3. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
4. Click "Create Key"
5. Copy the key (starts with `sk-or-v1-...`)
6. Add credits to your account at [openrouter.ai/credits](https://openrouter.ai/credits)

**Important**: Keep your API key secret! Never commit it to git or share it publicly.

---

## Understanding What You Need

### What is OpenRouter?

OpenRouter is like a "universal translator" for LLMs. Instead of having separate API keys for OpenAI, Google, Anthropic, etc., you use one OpenRouter key to access all of them. This simplifies the code significantly.

### What are the models being used?

By default, the council uses these models (configured in `backend/config.py`):

| Model | Provider | Role |
|-------|----------|------|
| `openai/gpt-5.1` | OpenAI | Council member |
| `google/gemini-3-pro-preview` | Google | Council member + Chairman |
| `anthropic/claude-sonnet-4.5` | Anthropic | Council member |
| `x-ai/grok-4` | xAI | Council member |

You can change these to any models available on OpenRouter.

---

## Step-by-Step Installation

### Step 1: Navigate to the Project Directory

```bash
cd llm-council-master
```

### Step 2: Create Your Environment File

Create a `.env` file in the project root to store your API key:

**On macOS/Linux:**
```bash
echo "OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here" > .env
```

**On Windows (PowerShell):**
```powershell
"OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here" | Out-File -FilePath .env -Encoding utf8
```

**Or manually:** Create a file named `.env` (no extension) with this content:
```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

### Step 3: Install Backend Dependencies

From the project root directory:

```bash
uv sync
```

This reads `pyproject.toml` and installs:
- FastAPI (web framework)
- uvicorn (web server)
- httpx (HTTP client for API calls)
- python-dotenv (reads .env file)
- pydantic (data validation)

### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

This installs React, Vite, and react-markdown.

### Step 5: Verify Installation

Your directory should now contain:

```
llm-council-master/
├── .env                    # Your API key (DO NOT COMMIT)
├── .venv/                  # Python virtual environment (created by uv)
├── frontend/
│   └── node_modules/       # JavaScript dependencies
└── ... (other files)
```

---

## Running the Application

The application has two parts that need to run simultaneously:
- **Backend**: Python FastAPI server (port 8001)
- **Frontend**: React development server (port 5173)

### Option A: Using the Start Script (macOS/Linux)

```bash
./start.sh
```

This starts both servers in background processes.

### Option B: Manual Start (Recommended for Windows or debugging)

**Open two terminal windows:**

**Terminal 1 - Backend:**
```bash
cd llm-council-master
uv run python -m backend.main
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx]
```

**Terminal 2 - Frontend:**
```bash
cd llm-council-master/frontend
npm run dev
```

You should see:
```
  VITE v7.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

### Step 6: Open in Browser

Navigate to: **http://localhost:5173**

You should see the LLM Council interface with a sidebar and chat area.

---

## Your First Query

1. Click **"New Conversation"** in the sidebar
2. Type a question in the text area at the bottom
3. Press **Enter** or click **Send**
4. Watch as the three stages complete:
   - **Stage 1**: Individual model responses (tabs to see each)
   - **Stage 2**: Peer rankings (see how models rated each other)
   - **Stage 3**: Final synthesized answer (green background)

### Example Questions to Try

- "What are the key differences between REST and GraphQL APIs?"
- "Explain quantum computing to a 10-year-old"
- "What's the best approach to learn a new programming language?"

---

## Customizing Models

### Changing Council Members

Edit `backend/config.py`:

```python
# backend/config.py

COUNCIL_MODELS = [
    "openai/gpt-5.1",              # Keep or change
    "google/gemini-3-pro-preview", # Keep or change
    "anthropic/claude-sonnet-4.5", # Keep or change
    "x-ai/grok-4",                 # Keep or change
]

CHAIRMAN_MODEL = "google/gemini-3-pro-preview"  # Who synthesizes the final answer
```

### Finding Model Names

Browse available models at: [openrouter.ai/models](https://openrouter.ai/models)

Model identifiers follow the format: `provider/model-name`

Examples:
- `openai/gpt-4o` - OpenAI GPT-4o
- `anthropic/claude-3-opus` - Claude 3 Opus
- `meta-llama/llama-3.1-405b-instruct` - Meta's Llama 3.1

### After Changing Models

Restart the backend server (Ctrl+C and run again) for changes to take effect.

---

## Understanding Costs

### How Pricing Works

OpenRouter charges per token (roughly 4 characters = 1 token):

1. **Input tokens**: Your question + context sent to each model
2. **Output tokens**: The model's response

Each query to LLM Council makes:
- **Stage 1**: N queries (N = number of council members, default 4)
- **Stage 2**: N queries (each model evaluates all responses)
- **Stage 3**: 1 query (chairman synthesizes)
- **Title generation**: 1 query (using a cheap model)

**Total: 2N + 2 API calls per user question** (default: 10 calls)

### Cost Estimation

Approximate costs per question (varies by model and response length):
- Budget models (Llama, Mistral): $0.01 - $0.05
- Mid-tier models (GPT-4o, Claude Sonnet): $0.10 - $0.50
- Premium models (GPT-5.1, Claude Opus): $0.50 - $2.00

**Tip**: Start with smaller/cheaper models while learning!

### Checking Your Usage

View your usage at: [openrouter.ai/activity](https://openrouter.ai/activity)

---

## Next Steps

Now that you have the application running:

1. **Understand the system**: Read [How It Works](./HOW_IT_WORKS.md)
2. **Learn the code structure**: Read [Architecture](./ARCHITECTURE.md)
3. **Add features**: Read [Extending the Codebase](./EXTENDING.md)

---

## Common Setup Issues

### "Module not found" errors

Make sure you're running from the project root:
```bash
cd llm-council-master
uv run python -m backend.main  # Note the -m flag
```

### CORS errors in browser

Ensure the backend is running on port 8001 and frontend on port 5173.

### API key not working

1. Check `.env` file exists in project root
2. Verify key starts with `sk-or-v1-`
3. Ensure you have credits on OpenRouter

For more issues, see [Troubleshooting](./TROUBLESHOOTING.md).
