# How LLM Council Works

This document explains the conceptual design and flow of LLM Council. Understanding this will help you make sense of the code and extend it effectively.

## Table of Contents

1. [The Problem Being Solved](#the-problem-being-solved)
2. [The 3-Stage Solution](#the-3-stage-solution)
3. [Stage 1: Collecting First Opinions](#stage-1-collecting-first-opinions)
4. [Stage 2: Anonymous Peer Review](#stage-2-anonymous-peer-review)
5. [Stage 3: Chairman Synthesis](#stage-3-chairman-synthesis)
6. [Why Anonymization Matters](#why-anonymization-matters)
7. [The Complete Data Flow](#the-complete-data-flow)
8. [Real Example Walkthrough](#real-example-walkthrough)

---

## The Problem Being Solved

When you ask a question to a single LLM, you get one perspective. That response might be:
- Incomplete (missing important information)
- Biased toward certain approaches
- Confidently wrong about some details
- Excellent in some areas but weak in others

Different LLMs have different training data, architectures, and "personalities." By consulting multiple models, we can:
- Get diverse perspectives
- Cross-validate information
- Identify consensus and disagreement
- Produce a more comprehensive final answer

**But there's a challenge**: How do we combine multiple responses effectively?

---

## The 3-Stage Solution

LLM Council uses a deliberation process inspired by academic peer review:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUESTION                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: FIRST OPINIONS                                         │
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Model A │  │ Model B │  │ Model C │  │ Model D │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│       ▼            ▼            ▼            ▼                  │
│  Response A   Response B   Response C   Response D              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PEER REVIEW (Anonymized!)                              │
│                                                                 │
│  Each model reviews ALL responses (labeled A, B, C, D)          │
│  without knowing which model wrote which response               │
│                                                                 │
│  Model A ranks: C > A > B > D                                   │
│  Model B ranks: C > B > A > D                                   │
│  Model C ranks: A > C > B > D                                   │
│  Model D ranks: C > A > D > B                                   │
│                                                                 │
│  Aggregate Ranking: C (best) → A → B → D (worst)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: CHAIRMAN SYNTHESIS                                     │
│                                                                 │
│  Chairman receives:                                             │
│  - Original question                                            │
│  - All Stage 1 responses (with model names)                     │
│  - All Stage 2 rankings (with model names)                      │
│                                                                 │
│  Chairman produces: Final synthesized answer                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL ANSWER                             │
│           (Displayed to user with green background)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Collecting First Opinions

### What Happens

1. Your question is sent to ALL council models simultaneously (in parallel)
2. Each model generates its response independently
3. All responses are collected and displayed in a tab view

### Code Location

`backend/council.py` → `stage1_collect_responses()`

### Key Design Decisions

- **Parallel execution**: All models are queried at the same time using `asyncio.gather()`. This is much faster than querying them one by one.

- **Graceful degradation**: If one model fails (times out, rate limited, etc.), the others continue. The system doesn't fail completely just because one model had an issue.

- **Simple prompt**: Models receive just the user's question. No special instructions are added at this stage.

### What You See in the UI

A tab interface showing each model's response. Click different tabs to see how each model answered.

---

## Stage 2: Anonymous Peer Review

### What Happens

1. All Stage 1 responses are **anonymized** as "Response A", "Response B", etc.
2. Each model receives a prompt asking it to evaluate and rank all responses
3. Models provide written evaluations and a final ranking
4. Rankings are parsed and aggregated

### Code Location

`backend/council.py` → `stage2_collect_rankings()`

### The Anonymization Process

```python
# Stage 1 produced these responses:
responses = {
    "openai/gpt-5.1": "GPT's answer...",
    "anthropic/claude-sonnet-4.5": "Claude's answer...",
    "google/gemini-3-pro-preview": "Gemini's answer...",
    "x-ai/grok-4": "Grok's answer...",
}

# Before Stage 2, they become:
anonymized = {
    "Response A": "GPT's answer...",      # Was GPT
    "Response B": "Claude's answer...",   # Was Claude
    "Response C": "Gemini's answer...",   # Was Gemini
    "Response D": "Grok's answer...",     # Was Grok
}

# A mapping is kept for later de-anonymization:
label_to_model = {
    "Response A": "openai/gpt-5.1",
    "Response B": "anthropic/claude-sonnet-4.5",
    "Response C": "google/gemini-3-pro-preview",
    "Response D": "x-ai/grok-4",
}
```

### The Ranking Prompt

Each model receives a carefully structured prompt:

```
You are evaluating different responses to the following question:

Question: {user's original question}

Here are the responses from different models (anonymized):

Response A:
{content of response A}

Response B:
{content of response B}

...

Your task:
1. First, evaluate each response individually. For each response, explain
   what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label

Example:
FINAL RANKING:
1. Response C
2. Response A
3. Response B
```

### Parsing the Rankings

The system extracts rankings using `parse_ranking_from_text()`:

1. Look for "FINAL RANKING:" section
2. Extract numbered list items like "1. Response C"
3. Return ordered list: `["Response C", "Response A", "Response B", "Response D"]`

### Calculating Aggregate Rankings

```python
# Example: 4 models each gave rankings
rankings = [
    ["Response C", "Response A", "Response B", "Response D"],  # Model 1
    ["Response C", "Response B", "Response A", "Response D"],  # Model 2
    ["Response A", "Response C", "Response B", "Response D"],  # Model 3
    ["Response C", "Response A", "Response D", "Response B"],  # Model 4
]

# Calculate average position (1-indexed):
# Response A: positions 2, 3, 1, 2 → average = 2.0
# Response B: positions 3, 2, 3, 4 → average = 3.0
# Response C: positions 1, 1, 2, 1 → average = 1.25 ← BEST
# Response D: positions 4, 4, 4, 3 → average = 3.75

# Final ranking: C (1.25) → A (2.0) → B (3.0) → D (3.75)
```

### What You See in the UI

- Tab view showing each model's full evaluation text
- "Extracted Ranking" section showing the parsed ranking list
- "Aggregate Rankings" section showing combined results ("Street Cred")

---

## Stage 3: Chairman Synthesis

### What Happens

1. A designated "chairman" model receives comprehensive context:
   - The original question
   - All Stage 1 responses (with real model names)
   - All Stage 2 evaluations (with real model names)
2. The chairman synthesizes a final answer

### Code Location

`backend/council.py` → `stage3_synthesize_final()`

### The Chairman Prompt

```
You are the Chairman of an LLM Council. Multiple AI models have provided
responses to a user's question, and then ranked each other's responses.

Original Question: {user's question}

STAGE 1 - Individual Responses:
Model: openai/gpt-5.1
Response: {GPT's response}

Model: anthropic/claude-sonnet-4.5
Response: {Claude's response}
...

STAGE 2 - Peer Rankings:
Model: openai/gpt-5.1
Ranking: {GPT's evaluation and ranking}
...

Your task as Chairman is to synthesize all of this information into a
single, comprehensive, accurate answer to the user's original question.
Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's
collective wisdom:
```

### What You See in the UI

The final synthesized answer with a green background, indicating it's the council's conclusion.

---

## Why Anonymization Matters

### The Bias Problem

Without anonymization, models might:

1. **Play favorites**: Prefer responses from certain providers
2. **Self-promote**: Rank their own response higher
3. **Brand bias**: Associate quality with familiar model names

### How Anonymization Helps

By using "Response A, B, C, D" instead of model names:
- Models judge responses purely on content quality
- No brand recognition bias
- More objective evaluation

### Transparency in the UI

The UI shows:
- Raw evaluation text with anonymized labels
- De-anonymized version (model names in **bold**) for user readability
- A note explaining that models received anonymous labels

---

## The Complete Data Flow

```
User types question
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Frontend (React)                                               │
│                                                               │
│ 1. User clicks send                                           │
│ 2. api.sendMessageStream() called                             │
│ 3. SSE connection established                                 │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ HTTP POST with SSE
        │
┌───────────────────────────────────────────────────────────────┐
│ Backend (FastAPI)                                              │
│                                                               │
│ POST /api/conversations/{id}/message/stream                   │
│                                                               │
│ 1. Save user message to storage                               │
│ 2. Start title generation (async, in background)              │
│ 3. Execute Stage 1 → stream results                           │
│ 4. Execute Stage 2 → stream results + metadata                │
│ 5. Execute Stage 3 → stream results                           │
│ 6. Save assistant message to storage                          │
│ 7. Send completion event                                      │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ During Stages 1, 2, 3
        │
┌───────────────────────────────────────────────────────────────┐
│ OpenRouter API                                                 │
│                                                               │
│ Routes requests to appropriate providers:                      │
│ - OpenAI                                                      │
│ - Google                                                      │
│ - Anthropic                                                   │
│ - xAI                                                         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ SSE events back to frontend
        │
┌───────────────────────────────────────────────────────────────┐
│ Frontend updates progressively                                 │
│                                                               │
│ stage1_start → Show "Stage 1 loading..."                      │
│ stage1_complete → Display Stage 1 tabs                        │
│ stage2_start → Show "Stage 2 loading..."                      │
│ stage2_complete → Display Stage 2 tabs + rankings             │
│ stage3_start → Show "Stage 3 loading..."                      │
│ stage3_complete → Display final answer                        │
│ title_complete → Update conversation title in sidebar         │
│ complete → Loading state off                                  │
└───────────────────────────────────────────────────────────────┘
```

### Title Generation (Parallel Process)

On the **first message** of a new conversation, a title is automatically generated:

- Runs **in parallel** with Stages 1-3 (doesn't slow down the response)
- Uses `google/gemini-2.5-flash` (a fast, inexpensive model)
- Generates a 3-5 word summary of the question
- Sent to frontend via `title_complete` SSE event
- Updates the conversation in the sidebar

This is why you see the conversation title change from "New Conversation" to something descriptive shortly after your first message.

---

## Real Example Walkthrough

### User Question

"What's the best way to learn Python?"

### Stage 1 Responses (Abbreviated)

**GPT-5.1**: "Start with the official Python tutorial, then build projects..."

**Claude Sonnet**: "Focus on fundamentals first: variables, loops, functions..."

**Gemini 3 Pro**: "I recommend a structured approach: online course + practice..."

**Grok 4**: "Dive into coding immediately. Pick a project you're excited about..."

### Stage 2 Evaluations (Abbreviated)

**GPT-5.1 evaluates:**
> Response A provides solid advice with specific resources...
> Response B focuses well on fundamentals but lacks project guidance...
> Response C offers good structure but is somewhat generic...
> Response D is motivating but may frustrate beginners...
>
> FINAL RANKING:
> 1. Response A
> 2. Response C
> 3. Response B
> 4. Response D

**Other models similarly evaluate and rank...**

### Stage 2 Aggregate Result

| Rank | Model | Average Position |
|------|-------|------------------|
| 1 | GPT-5.1 | 1.5 |
| 2 | Claude Sonnet | 2.0 |
| 3 | Gemini 3 Pro | 2.75 |
| 4 | Grok 4 | 3.75 |

### Stage 3 Synthesis (Abbreviated)

> Based on the council's deliberations, here's the recommended approach to learning Python:
>
> **1. Start with Fundamentals** (emphasized by Claude and Gemini)
> Begin with core concepts: variables, data types, loops, and functions...
>
> **2. Use Quality Resources** (highlighted by GPT)
> The official Python tutorial is excellent, supplemented by...
>
> **3. Build Projects** (consensus across all models)
> Apply what you learn through hands-on projects...
>
> **4. Stay Motivated** (Grok's valuable insight)
> Choose projects that genuinely interest you...

---

## Key Takeaways

1. **Parallel execution** makes the system fast despite multiple API calls
2. **Anonymization** ensures fair, unbiased peer review
3. **Structured prompts** enable reliable parsing of rankings
4. **Progressive streaming** provides good user experience during long operations
5. **Graceful degradation** keeps the system working even if some models fail

---

## Next Steps

- **See the code**: [Architecture](./ARCHITECTURE.md) and [Backend Guide](./BACKEND_GUIDE.md)
- **Modify the flow**: [Extending the Codebase](./EXTENDING.md)
