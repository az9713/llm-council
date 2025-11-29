# Understanding OpenRouter

This guide explains what OpenRouter is and how LLM Council uses it to orchestrate multiple AI models.

## Table of Contents

1. [What is OpenRouter?](#what-is-openrouter)
2. [The Problem OpenRouter Solves](#the-problem-openrouter-solves)
3. [How OpenRouter Works](#how-openrouter-works)
4. [Why LLM Council Uses OpenRouter](#why-llm-council-uses-openrouter)
5. [How LLM Council Uses OpenRouter](#how-llm-council-uses-openrouter)
6. [Setting Up OpenRouter](#setting-up-openrouter)
7. [Understanding Costs](#understanding-costs)
8. [Available Models](#available-models)
9. [Common Questions](#common-questions)

---

## What is OpenRouter?

**OpenRouter** is a unified API gateway that provides access to AI models from multiple providers through a single interface.

Think of it like this:

```
WITHOUT OpenRouter:
┌─────────────┐     ┌─────────────────┐
│ Your App    │────▶│ OpenAI API      │  (different API format)
│             │────▶│ Anthropic API   │  (different API format)
│             │────▶│ Google AI API   │  (different API format)
│             │────▶│ xAI API         │  (different API format)
└─────────────┘     └─────────────────┘
   You manage 4 different API keys, 4 different formats, 4 different billing accounts

WITH OpenRouter:
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│ Your App    │────▶│ OpenRouter  │────▶│ OpenAI          │
│             │     │   (router)  │────▶│ Anthropic       │
│             │     │             │────▶│ Google          │
│             │     │             │────▶│ xAI             │
└─────────────┘     └─────────────┘     └─────────────────┘
   You manage 1 API key, 1 format, 1 billing account
```

**Key point**: OpenRouter is not an AI model itself. It's a middleman that routes your requests to the actual AI providers.

---

## The Problem OpenRouter Solves

### Without OpenRouter

If you wanted to use GPT-4, Claude, and Gemini in the same application, you would need to:

1. **Create accounts** with OpenAI, Anthropic, and Google
2. **Get API keys** from each provider
3. **Learn different APIs** - each provider has slightly different request/response formats
4. **Manage billing** across multiple platforms
5. **Handle rate limits** differently for each provider
6. **Write adapter code** to normalize responses

### With OpenRouter

1. **One account** at openrouter.ai
2. **One API key** that works for all models
3. **One API format** (OpenAI-compatible) for all models
4. **One billing system** with unified usage tracking
5. **Consistent behavior** across providers

---

## How OpenRouter Works

### The API Format

OpenRouter uses the **OpenAI Chat Completions format**, which has become an industry standard:

```bash
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer YOUR_OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

The response follows the same format:

```json
{
  "id": "gen-abc123",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      }
    }
  ],
  "model": "openai/gpt-4o",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15
  }
}
```

### Model Identifiers

OpenRouter uses a `provider/model-name` format:

| Model | OpenRouter Identifier |
|-------|----------------------|
| GPT-4o | `openai/gpt-4o` |
| GPT-4o Mini | `openai/gpt-4o-mini` |
| Claude 3.5 Sonnet | `anthropic/claude-3.5-sonnet` |
| Claude 3 Opus | `anthropic/claude-3-opus` |
| Gemini 1.5 Pro | `google/gemini-pro-1.5` |
| Gemini Flash | `google/gemini-flash-1.5` |
| Grok | `x-ai/grok-beta` |
| Llama 3.1 405B | `meta-llama/llama-3.1-405b-instruct` |

Full list: [openrouter.ai/models](https://openrouter.ai/models)

### What Happens Behind the Scenes

1. You send a request to OpenRouter with `model: "anthropic/claude-3.5-sonnet"`
2. OpenRouter validates your API key and checks your credits
3. OpenRouter translates your request to Anthropic's native format
4. OpenRouter sends the request to Anthropic's API
5. Anthropic processes the request and returns a response
6. OpenRouter translates the response back to the standard format
7. You receive the response

**Latency overhead**: OpenRouter adds minimal latency (typically <100ms) for this routing.

---

## Why LLM Council Uses OpenRouter

LLM Council needs to query **multiple different AI models** and compare their responses. OpenRouter is the ideal solution for this use case:

### 1. Simplified Multi-Model Architecture

```python
# WITHOUT OpenRouter - you'd need separate clients for each provider:
openai_response = openai_client.chat.completions.create(...)
anthropic_response = anthropic_client.messages.create(...)  # Different API!
google_response = google_client.generate_content(...)       # Different API!

# WITH OpenRouter - one function works for all:
gpt_response = await query_model("openai/gpt-4o", messages)
claude_response = await query_model("anthropic/claude-3.5-sonnet", messages)
gemini_response = await query_model("google/gemini-pro-1.5", messages)
```

### 2. Easy Model Swapping

Adding or removing models from the council requires changing just one line:

```python
# backend/config.py
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
    # Just add a new line to add a new model!
    "meta-llama/llama-3.1-405b-instruct",
]
```

### 3. Parallel Queries Made Simple

Since all models use the same API format, parallel querying is straightforward:

```python
# All models queried simultaneously with identical code
tasks = [query_model(model, messages) for model in COUNCIL_MODELS]
responses = await asyncio.gather(*tasks)
```

### 4. Graceful Degradation

If one provider has issues, others continue working:

```python
# If Anthropic is down, OpenAI and Google responses still come through
# The system continues with available responses
```

### 5. Cost Tracking

One dashboard shows usage across all models, making it easy to understand the cost of running the council.

---

## How LLM Council Uses OpenRouter

### File: `backend/config.py`

Defines the OpenRouter configuration:

```python
# The API key (stored in .env file)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# The endpoint (standard for all requests)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# The models to use in the council
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]

# The model that synthesizes the final answer
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"
```

### File: `backend/openrouter.py`

Contains two key functions:

#### `query_model()` - Query a Single Model

```python
async def query_model(
    model: str,                        # e.g., "openai/gpt-4o"
    messages: List[Dict[str, str]],    # [{"role": "user", "content": "..."}]
    timeout: float = 120.0             # Wait up to 2 minutes
) -> Optional[Dict[str, Any]]:
```

This function:
1. Builds the HTTP request with authorization header
2. Sends POST request to OpenRouter
3. Parses the response
4. Returns `{'content': '...'}` or `None` if failed

**Error handling**: If a request fails (timeout, rate limit, etc.), the function returns `None` instead of crashing. This enables graceful degradation.

#### `query_models_parallel()` - Query Multiple Models Simultaneously

```python
async def query_models_parallel(
    models: List[str],                 # ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
```

This function:
1. Creates async tasks for each model
2. Uses `asyncio.gather()` to run all tasks concurrently
3. Returns a dictionary mapping each model to its response

**Performance benefit**: If each model takes 5 seconds, querying 4 models sequentially takes 20 seconds. In parallel, it takes ~5 seconds.

### How Stages Use OpenRouter

**Stage 1** - Collect individual responses:
```python
# council.py
responses = await query_models_parallel(COUNCIL_MODELS, messages)
# Returns: {"openai/gpt-5.1": {...}, "anthropic/claude-sonnet-4.5": {...}, ...}
```

**Stage 2** - Collect peer rankings:
```python
# council.py
# Same models, but with a different prompt (the ranking prompt)
rankings = await query_models_parallel(COUNCIL_MODELS, ranking_messages)
```

**Stage 3** - Chairman synthesis:
```python
# council.py
final = await query_model(CHAIRMAN_MODEL, synthesis_messages)
```

**Title Generation** - Generate conversation title:
```python
# council.py
# Uses a fast, cheap model (hardcoded)
title = await query_model("google/gemini-2.5-flash", title_messages)
```

---

## Setting Up OpenRouter

### Step 1: Create an Account

1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up with email or OAuth (Google, GitHub)

### Step 2: Add Credits

1. Go to [openrouter.ai/credits](https://openrouter.ai/credits)
2. Add funds (minimum $5)
3. Consider enabling auto-refill to avoid interruptions

### Step 3: Get Your API Key

1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Click "Create Key"
3. Give it a name (e.g., "llm-council")
4. Copy the key (it starts with `sk-or-v1-`)

### Step 4: Configure LLM Council

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Step 5: Verify Setup

```bash
# Quick test
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer sk-or-v1-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}'
```

---

## Understanding Costs

### How Pricing Works

OpenRouter charges based on **tokens** (roughly 4 characters = 1 token):
- **Input tokens**: What you send to the model
- **Output tokens**: What the model generates

Each model has different pricing. Example rates (as of 2024):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o Mini | $0.15 | $0.60 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Gemini 1.5 Flash | $0.075 | $0.30 |
| Llama 3.1 405B | $3.00 | $3.00 |

*Prices change frequently - check [openrouter.ai/models](https://openrouter.ai/models) for current rates.*

### LLM Council Cost Breakdown

For each user query, LLM Council makes:

| Stage | API Calls | Models Used |
|-------|-----------|-------------|
| Stage 1 | 4 calls | All 4 council models |
| Stage 2 | 4 calls | All 4 council models |
| Stage 3 | 1 call | Chairman model |
| Title (first message only) | 1 call | Gemini Flash |

**Total per query**: 9 API calls (10 on first message)

### Estimating Costs

A typical query might use:
- ~100 input tokens per Stage 1 call (the question)
- ~500 output tokens per Stage 1 call (the answer)
- ~3000 input tokens per Stage 2 call (question + all Stage 1 responses)
- ~300 output tokens per Stage 2 call (evaluation)
- ~5000 input tokens for Stage 3 (everything combined)
- ~500 output tokens for Stage 3 (final answer)

**Rough estimate**: $0.05 - $0.15 per query depending on models used.

### Reducing Costs

1. **Use cheaper models**: Replace expensive models with cheaper alternatives
   ```python
   COUNCIL_MODELS = [
       "openai/gpt-4o-mini",        # Instead of gpt-4o
       "google/gemini-flash-1.5",   # Instead of gemini-pro
       "anthropic/claude-3-haiku",  # Instead of claude-3.5-sonnet
   ]
   ```

2. **Reduce council size**: Use 2-3 models instead of 4

3. **Monitor usage**: Check [openrouter.ai/activity](https://openrouter.ai/activity)

---

## Available Models

OpenRouter provides access to 100+ models. Here are categories relevant to LLM Council:

### Flagship Models (Best Quality)

| Model | Identifier | Best For |
|-------|-----------|----------|
| GPT-4o | `openai/gpt-4o` | General purpose, coding |
| Claude 3.5 Sonnet | `anthropic/claude-3.5-sonnet` | Analysis, writing |
| Gemini 1.5 Pro | `google/gemini-pro-1.5` | Long context, reasoning |
| Grok | `x-ai/grok-beta` | Current events, casual |

### Fast & Cheap Models (Good for Title Generation)

| Model | Identifier | Notes |
|-------|-----------|-------|
| GPT-4o Mini | `openai/gpt-4o-mini` | Fast, very capable |
| Gemini 1.5 Flash | `google/gemini-flash-1.5` | Extremely fast |
| Claude 3 Haiku | `anthropic/claude-3-haiku` | Quick responses |

### Open Source Models

| Model | Identifier | Notes |
|-------|-----------|-------|
| Llama 3.1 405B | `meta-llama/llama-3.1-405b-instruct` | Largest open model |
| Llama 3.1 70B | `meta-llama/llama-3.1-70b-instruct` | Good balance |
| Mixtral 8x22B | `mistralai/mixtral-8x22b-instruct` | Fast, capable |

### Finding Models

1. Browse: [openrouter.ai/models](https://openrouter.ai/models)
2. Filter by: price, context length, capabilities
3. Check availability (some models have waitlists)
4. Copy the model identifier for use in `config.py`

---

## Common Questions

### Is OpenRouter free?

No. You pay for the tokens used, similar to using the providers directly. OpenRouter adds a small markup (typically 0-20%) on top of provider prices. The convenience of unified access is worth the small premium for most use cases.

### Is my data safe?

OpenRouter acts as a pass-through. Your prompts go to the actual providers (OpenAI, Anthropic, etc.) who have their own privacy policies. OpenRouter doesn't train on your data. Review their [privacy policy](https://openrouter.ai/privacy) for details.

### What if OpenRouter goes down?

If OpenRouter has an outage, all model requests will fail. This is a trade-off of using a unified gateway. For production systems requiring high availability, you might want fallback logic to call providers directly.

### Can I use my own API keys?

OpenRouter's value is the unified billing and API. If you have direct API keys with providers, you could modify `backend/openrouter.py` to call them directly, but you'd lose the benefits of unified access.

### Why do some models show as unavailable?

Models can be temporarily unavailable due to:
- Provider outages
- Rate limiting
- Model deprecation
- Waitlist restrictions

The council continues working with available models (graceful degradation).

### How do I know which models are best?

LLM Council itself helps answer this! The Stage 2 rankings show which models the council collectively prefers. Over time, you can observe patterns and adjust your model selection.

---

## Summary

OpenRouter is the **backbone** of LLM Council's multi-model architecture. It provides:

- **Unified API**: One format for all models
- **Single billing**: One account, one key, one dashboard
- **Easy scaling**: Add models with one line of config
- **Graceful degradation**: Failed models don't crash the system

Without OpenRouter (or a similar gateway), LLM Council would need significant additional code to handle multiple provider APIs, making it much more complex to build and maintain.

---

## Next Steps

- Set up OpenRouter: [Setting Up OpenRouter](#setting-up-openrouter)
- Configure models: [Backend Guide](./BACKEND_GUIDE.md#configpy---configuration)
- Understand the full flow: [How It Works](./HOW_IT_WORKS.md)
