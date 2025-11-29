# LLM Council - Quick Start Guide with Hands-On Use Cases

Welcome! This guide will get you up and running with LLM Council and show you **10 practical use cases** that demonstrate the power of having multiple AI models collaborate on your questions.

Each use case is designed to be a "quick win" - you'll see real results in minutes while learning the system's capabilities.

---

## Table of Contents

1. [Complete Setup Guide](#complete-setup-guide)
2. [Quick Start Checklist](#quick-start-checklist)
3. [Use Cases (Ranked by Complexity)](#use-cases-ranked-by-complexity)
   - [Use Case 1: Your First Council Query (Easiest)](#use-case-1-your-first-council-query)
   - [Use Case 2: Compare Different Perspectives](#use-case-2-compare-different-perspectives)
   - [Use Case 3: Export and Share Conversations](#use-case-3-export-and-share-conversations)
   - [Use Case 4: Watch the AI Think in Real-Time](#use-case-4-watch-the-ai-think-in-real-time)
   - [Use Case 5: Customize AI Behavior with System Prompts](#use-case-5-customize-ai-behavior-with-system-prompts)
   - [Use Case 6: Track Performance Over Time](#use-case-6-track-performance-over-time)
   - [Use Case 7: Enable Deep Reasoning (Chain-of-Thought)](#use-case-7-enable-deep-reasoning-chain-of-thought)
   - [Use Case 8: Multi-Chairman Consensus](#use-case-8-multi-chairman-consensus)
   - [Use Case 9: Save Money with Smart Caching](#use-case-9-save-money-with-smart-caching)
   - [Use Case 10: Debate Mode - AI vs AI](#use-case-10-debate-mode---ai-vs-ai)
4. [What's Next?](#whats-next)

---

## Complete Setup Guide

Before trying the use cases, let's get the application running. **Follow every step carefully.**

### Prerequisites - What You Need

| Requirement | Why You Need It | How to Check |
|-------------|-----------------|--------------|
| **Python 3.10+** | Runs the backend server | `python --version` |
| **Node.js 18+** | Runs the frontend UI | `node --version` |
| **uv** | Installs Python packages fast | `uv --version` |
| **OpenRouter API Key** | Accesses AI models | Sign up at openrouter.ai |

### Step 1: Install Python (if not installed)

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.12 or newer
3. Run the installer
4. **IMPORTANT**: Check "Add Python to PATH" during installation
5. Restart your terminal

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.12
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**Verify installation:**
```bash
python --version
# Expected output: Python 3.10.x or higher
```

### Step 2: Install Node.js (if not installed)

**Windows:**
1. Go to https://nodejs.org/
2. Download the LTS version (18.x or higher)
3. Run the installer with default settings
4. Restart your terminal

**macOS:**
```bash
brew install node
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

**Verify installation:**
```bash
node --version
# Expected output: v18.x.x or higher

npm --version
# Expected output: 9.x.x or higher
```

### Step 3: Install uv (Python package manager)

uv is a modern, fast Python package manager. It's required for this project.

**Windows (PowerShell - Run as Administrator):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then **restart your terminal** (close and reopen).

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal or run:
```bash
source $HOME/.local/bin/env
```

**Verify installation:**
```bash
uv --version
# Expected output: uv 0.x.x
```

### Step 4: Get Your OpenRouter API Key

OpenRouter provides access to multiple AI models (GPT-4, Claude, Gemini, etc.) through a single API.

1. **Go to:** https://openrouter.ai/
2. **Click** "Sign Up" (top right)
3. **Create an account** (you can use Google/GitHub login)
4. **Go to:** https://openrouter.ai/keys
5. **Click** "Create Key"
6. **Copy** the key (it looks like `sk-or-v1-abc123...`)
7. **Add credits:** https://openrouter.ai/credits
   - Start with $5-10 for testing
   - Each query costs ~$0.01-$0.50 depending on models

**IMPORTANT:** Keep your API key secret! Never share it or commit it to Git.

### Step 5: Navigate to the Project Directory

Open a terminal and navigate to where you extracted/cloned the project:

**Windows (Command Prompt or PowerShell):**
```cmd
cd C:\Users\YourName\Downloads\llm-council-master
```

**macOS/Linux:**
```bash
cd ~/Downloads/llm-council-master
```

### Step 6: Create the Environment File

This stores your secret API key.

**Windows (PowerShell):**
```powershell
# Replace YOUR_KEY with your actual OpenRouter API key
"OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE" | Out-File -FilePath .env -Encoding ASCII
```

**Windows (Command Prompt):**
```cmd
echo OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE > .env
```

**macOS/Linux:**
```bash
echo "OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE" > .env
```

**Verify the file was created:**
```bash
# Windows
type .env

# macOS/Linux
cat .env
```

You should see your API key printed.

### Step 7: Install Backend Dependencies

From the project root directory:

```bash
uv sync
```

**What this does:**
- Creates a virtual environment (`.venv/` folder)
- Installs: FastAPI, uvicorn, httpx, python-dotenv, pydantic

**Expected output:**
```
Resolved 20 packages in 2.5s
Downloaded 20 packages in 1.2s
Installed 20 packages in 0.8s
```

### Step 8: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

**What this does:**
- Creates `node_modules/` folder with JavaScript packages
- Installs: React, Vite, react-markdown

**Expected output:**
```
added 150 packages in 30s
```

### Step 9: Start the Application

You need **two terminal windows** running simultaneously.

**Terminal 1 - Start the Backend:**

```bash
# Make sure you're in the project root directory
uv run python -m backend.main
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

**Leave this terminal running!**

**Terminal 2 - Start the Frontend:**

Open a NEW terminal window, navigate to the project, then:

```bash
cd frontend
npm run dev
```

**Expected output:**
```
  VITE v7.x.x  ready in 300 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

**Leave this terminal running too!**

### Step 10: Open the Application

Open your web browser and go to:

**http://localhost:5173**

You should see the LLM Council interface with:
- A sidebar on the left (conversations list)
- A main chat area on the right
- A text input at the bottom

**Congratulations! You're ready to try the use cases!**

---

## Quick Start Checklist

Before proceeding, verify everything is working:

- [ ] Python 3.10+ installed (`python --version`)
- [ ] Node.js 18+ installed (`node --version`)
- [ ] uv installed (`uv --version`)
- [ ] `.env` file created with your API key
- [ ] Backend running (Terminal 1 shows "Uvicorn running")
- [ ] Frontend running (Terminal 2 shows "VITE ready")
- [ ] Browser shows the LLM Council interface at http://localhost:5173

---

## Use Cases (Ranked by Complexity)

Each use case includes:
- **Motivation**: Why this is useful
- **Time to Complete**: How long it takes
- **Features Used**: Which LLM Council features you'll learn
- **Step-by-Step Instructions**: Exactly what to do

---

## Use Case 1: Your First Council Query

**Complexity: ★☆☆☆☆ (Easiest)**

### Motivation

See how multiple AI models answer the same question and collectively arrive at a better answer than any single model could provide. This is the core value of LLM Council.

### Time to Complete: 2 minutes

### Features Used
- Basic council deliberation
- Stage 1 (model responses)
- Stage 2 (peer ranking)
- Stage 3 (synthesis)

### Step-by-Step Instructions

1. **Click "New Conversation"** in the left sidebar

2. **Type this question** in the text area at the bottom:
   ```
   What are three practical tips for improving sleep quality?
   ```

3. **Press Enter** or click the Send button

4. **Watch the magic happen:**
   - **Stage 1** appears: Click each tab (labeled with model names) to see how each AI answered your question. Notice how they give different tips and explanations.

   - **Stage 2** appears: This shows how each model ranked the other responses. The aggregate rankings at the top show which response was considered best overall.

   - **Stage 3** appears (green background): This is the Chairman's synthesis - a unified answer that combines the best insights from all models.

### What to Notice

- Different models emphasize different aspects (some focus on habits, others on environment)
- The rankings show which responses the council found most valuable
- The final synthesis is more comprehensive than any single response

### Quick Win Achieved!

You've just experienced the core feature: getting multiple AI perspectives synthesized into one superior answer.

---

## Use Case 2: Compare Different Perspectives

**Complexity: ★☆☆☆☆ (Easy)**

### Motivation

Some questions have multiple valid viewpoints. The council excels at presenting these fairly because anonymous peer review prevents bias.

### Time to Complete: 3 minutes

### Features Used
- Multi-perspective analysis
- Anonymous peer review
- Consensus building

### Step-by-Step Instructions

1. **Click "New Conversation"**

2. **Ask a question with multiple valid perspectives:**
   ```
   What are the pros and cons of remote work vs office work?
   ```

3. **Press Enter** and wait for all stages

4. **In Stage 1, click through each model's tab:**
   - Notice how some models might favor remote work
   - Others might emphasize office benefits
   - Each brings unique considerations

5. **In Stage 2, observe the rankings:**
   - Look at which response was ranked highest
   - Read the evaluation comments - models explain why they ranked others the way they did

6. **In Stage 3, read the synthesis:**
   - See how the Chairman balanced all perspectives
   - Notice it addresses both sides fairly

### What to Notice

The synthesis doesn't just pick a "winner" - it incorporates valuable points from all responses, creating a balanced view that a single model might not achieve.

### Quick Win Achieved!

You've seen how the council handles nuanced topics with multiple valid viewpoints.

---

## Use Case 3: Export and Share Conversations

**Complexity: ★☆☆☆☆ (Easy)**

### Motivation

Save your council conversations for later reference, share them with colleagues, or use them in reports. Export as Markdown (human-readable) or JSON (machine-readable).

### Time to Complete: 2 minutes

### Features Used
- Conversation export
- Markdown format
- JSON format

### Step-by-Step Instructions

1. **Have a conversation with at least one message** (use an existing one from previous use cases or create a new one)

2. **Look at the top of the chat area** - you should see:
   - "Export MD" button
   - "Export JSON" button
   - (These only appear when there are messages)

3. **Click "Export MD":**
   - A file downloads to your computer
   - Open it in any text editor or Markdown viewer
   - Notice it contains all three stages formatted nicely

4. **Click "Export JSON":**
   - A file downloads
   - Open it in a text editor
   - Notice the structured data format

5. **Try opening the Markdown file** in:
   - VS Code (shows formatted preview)
   - GitHub (renders beautifully)
   - Notion (paste it in)
   - Any Markdown viewer

### What to Notice

The Markdown export preserves:
- The original question
- Each model's response (Stage 1)
- Rankings and evaluations (Stage 2)
- The final synthesis (Stage 3)

### Quick Win Achieved!

You can now save and share any council conversation in a professional format.

---

## Use Case 4: Watch the AI Think in Real-Time

**Complexity: ★★☆☆☆ (Beginner-Friendly)**

### Motivation

Instead of waiting for complete responses, watch AI models generate their answers token-by-token in real-time. Plus, see what's happening "behind the scenes" with the Process Monitor.

### Time to Complete: 3 minutes

### Features Used
- Streaming responses
- Live Process Monitor
- Verbosity levels

### Step-by-Step Instructions

1. **Look at the settings bar** (top of the chat area)
   - Find the **"Process"** button
   - Click it to open the Process Monitor side panel

2. **In the Process Monitor panel:**
   - You'll see a verbosity slider (0-3)
   - Set it to **3 (Verbose)** for maximum detail

3. **Click "New Conversation"**

4. **Ask a question that requires a longer response:**
   ```
   Explain the concept of machine learning to someone who has never heard of it. Include an analogy.
   ```

5. **Press Enter and watch:**

   **In the main chat area:**
   - Stage 1 responses appear character-by-character
   - Click different model tabs to see them all streaming
   - Notice the pulsing blue dot on tabs that are still generating

   **In the Process Monitor:**
   - Events appear in real-time with timestamps
   - Blue events = stage transitions
   - Purple events = model operations
   - Green events = successes
   - Teal events = data/statistics

6. **Try different verbosity levels:**
   - **0 (Silent)**: No process events
   - **1 (Basic)**: Only stage transitions
   - **2 (Standard)**: Stage + model events
   - **3 (Verbose)**: Everything including statistics

### What to Notice

- Streaming shows you're not just waiting - work is happening
- The Process Monitor reveals the complexity of the council deliberation
- Different models have different response speeds

### Quick Win Achieved!

You've experienced real-time AI generation and learned to monitor the council's internal process.

---

## Use Case 5: Customize AI Behavior with System Prompts

**Complexity: ★★☆☆☆ (Beginner-Friendly)**

### Motivation

Tell all council members how to behave. Make them respond as experts, adjust their tone, or focus on specific aspects. The system prompt is instructions given to every model before your question.

### Time to Complete: 5 minutes

### Features Used
- Custom system prompts
- Settings panel
- Behavioral customization

### Step-by-Step Instructions

1. **Find the settings icon** (gear icon) in the settings bar
   - Click it to expand the settings panel

2. **Locate the "System Prompt" textarea**

3. **Enter a system prompt - try this example:**
   ```
   You are a patient teacher explaining concepts to a curious 10-year-old. Use simple language, fun analogies, and avoid jargon. Be encouraging and end with an interesting fact.
   ```

4. **Notice the blue dot indicator** in the settings bar
   - This shows a system prompt is active

5. **Click "New Conversation"**

6. **Ask a complex question:**
   ```
   How does the internet work?
   ```

7. **Compare the responses:**
   - Notice simpler language
   - Look for analogies and fun facts
   - See how the tone is encouraging

8. **Try a different system prompt:**
   ```
   You are a senior software engineer conducting a technical interview. Give precise, technical answers. Include code examples where relevant.
   ```

9. **Ask the same question** about the internet
   - See how dramatically different the responses are!

10. **To remove the system prompt:**
    - Clear the textarea
    - The blue dot disappears

### What to Notice

The same question gets completely different responses based on the system prompt. This is powerful for:
- Teaching different audiences
- Technical vs. non-technical explanations
- Different tones (formal, casual, humorous)

### Quick Win Achieved!

You can now customize how all council members behave for any use case.

---

## Use Case 6: Track Performance Over Time

**Complexity: ★★☆☆☆ (Beginner-Friendly)**

### Motivation

See which AI models consistently give the best answers. The Performance Dashboard tracks win rates, rankings, and costs across all your queries.

### Time to Complete: 5 minutes

### Features Used
- Performance Dashboard
- Analytics tracking
- Model statistics
- Cost monitoring

### Step-by-Step Instructions

1. **First, run at least 3-5 queries** to generate data
   - Use previous use cases or ask new questions
   - Each query contributes to the statistics

2. **Find the "Dashboard" button** in the settings bar
   - It has a purple/indigo theme
   - Click it to open the Performance Dashboard

3. **Explore the Leaderboard tab:**
   - Models are ranked by win rate
   - Top 3 have medal icons (🥇🥈🥉)
   - See which models are consistently ranked #1

4. **Click the "Model Details" tab:**
   - Each model has a detailed card
   - See rank distribution (how often each model gets 1st, 2nd, 3rd place)
   - View total tokens used and costs

5. **Click the "Chairman Stats" tab:**
   - See how often each model has been chairman
   - View chairman-specific costs

6. **Check the Summary Statistics** (top of dashboard):
   - Total queries processed
   - Number of unique models used
   - Date range of your data

7. **Use the Refresh button** to update data after new queries

8. **Optional: Clear data** (bottom of dashboard)
   - Click to reset all analytics
   - Useful for fresh starts

### What to Notice

- Some models consistently outperform others for certain question types
- You can see exactly how much you've spent
- Rankings help you decide which models to keep or replace

### Quick Win Achieved!

You now have data-driven insights into which AI models work best for your needs.

---

## Use Case 7: Enable Deep Reasoning (Chain-of-Thought)

**Complexity: ★★★☆☆ (Intermediate)**

### Motivation

Force ALL models to show their thinking process step-by-step: THINKING → ANALYSIS → CONCLUSION. This makes responses more transparent and helps identify the best reasoning, not just the best-sounding answer.

### Time to Complete: 5 minutes

### Features Used
- Chain-of-Thought (CoT) mode
- Structured reasoning
- ReasoningView component

### Step-by-Step Instructions

1. **Open the settings panel** (gear icon)

2. **Find the "Chain-of-Thought" toggle**
   - It should say "CoT" or have a toggle switch
   - Turn it ON

3. **Notice the "CoT" badge** appears in the settings bar
   - This indicates Chain-of-Thought mode is active

4. **Click "New Conversation"**

5. **Ask a question that benefits from reasoning:**
   ```
   A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?
   ```
   (This is a classic trick question - watch how models reason through it!)

6. **In Stage 1, click a model's tab:**
   - You'll see a structured response with three sections:
   - **Step 1 - Thinking** (blue): Initial thoughts
   - **Step 2 - Analysis** (purple): Working through the problem
   - **Step 3 - Conclusion** (green): Final answer

7. **Click the expand/collapse buttons** on each section
   - Conclusion is expanded by default
   - Click to reveal Thinking and Analysis

8. **Compare reasoning across models:**
   - Some might fall for the trick question
   - Others will correctly identify it
   - The reasoning shows WHY they got it right or wrong

9. **Try a more complex question:**
   ```
   Should I learn Python or JavaScript first as my first programming language? Consider my goal is to become a full-stack web developer.
   ```

10. **In Stage 2**, notice rankings now evaluate:
    - Reasoning quality (not just the answer)
    - Analysis depth
    - Conclusion accuracy

### What to Notice

- CoT exposes HOW models think, not just WHAT they conclude
- You can identify models with better reasoning skills
- Stage 2 rankings become more meaningful because they evaluate the thinking process

### Quick Win Achieved!

You can now see inside AI reasoning and make better judgments about response quality.

---

## Use Case 8: Multi-Chairman Consensus

**Complexity: ★★★☆☆ (Intermediate)**

### Motivation

Instead of one AI creating the final synthesis, have MULTIPLE chairmen create independent syntheses, then a "supreme chairman" picks the best one. This provides diversity in synthesis approaches and quality assurance.

### Time to Complete: 5 minutes

### Features Used
- Multi-Chairman mode
- Ensemble synthesis
- Supreme chairman selection

### Step-by-Step Instructions

1. **Open the settings panel**

2. **Find "Multi-Chairman" toggle**
   - Turn it ON
   - Notice the "MC" badge appears in settings bar

3. **Click "New Conversation"**

4. **Ask a question that benefits from synthesis:**
   ```
   What are the most important factors to consider when choosing a career path?
   ```

5. **Watch Stage 3 closely:**
   - Instead of one response, you'll see multiple tabs (A, B, C...)
   - Each tab is a different chairman's synthesis
   - Click through to compare them

6. **Look for the selection section:**
   - Shows which synthesis was chosen (checkmark on tab)
   - Click "Show Evaluation Details" to see WHY it was chosen
   - The supreme chairman explains its reasoning

7. **Compare the syntheses:**
   - Notice different organizational approaches
   - Some might be more comprehensive
   - Others might be more concise
   - The selected one balances quality factors

8. **Try with a technical question:**
   ```
   Explain the differences between SQL and NoSQL databases and when to use each.
   ```

9. **Compare which chairman handles technical topics better**

### What to Notice

- Different chairmen have different synthesis styles
- The selection process adds a quality check
- You get multiple "drafts" of the final answer to compare

### Quick Win Achieved!

You've experienced ensemble AI synthesis - multiple AIs collaborating on the final answer.

---

## Use Case 9: Save Money with Smart Caching

**Complexity: ★★★☆☆ (Intermediate)**

### Motivation

If you ask similar questions, why pay for the same AI work twice? Semantic caching stores answers and returns cached results for similar queries - saving time and money.

### Time to Complete: 5 minutes

### Features Used
- Response caching
- Semantic similarity
- Cache statistics

### Step-by-Step Instructions

1. **Open the settings panel**

2. **Find "Response Cache" toggle**
   - Turn it ON
   - Notice the "CA" badge (green) appears in settings bar

3. **Click "New Conversation"**

4. **Ask a question:**
   ```
   What are the health benefits of drinking green tea?
   ```

5. **Wait for the full response** (all three stages)

6. **Now ask a SIMILAR question** (new conversation or same):
   ```
   What health benefits does green tea provide?
   ```

7. **Notice the response is INSTANT:**
   - The Process Monitor (if open) shows "Cache Hit"
   - The response is identical to before
   - No API calls were made!

8. **Check cache statistics:**
   - In the settings or via API, you can see:
   - Total cache hits
   - Total cache misses
   - Estimated cost saved

9. **Try a DIFFERENT question:**
   ```
   What are the health benefits of black coffee?
   ```

10. **This will be a cache MISS:**
    - Similar topic but different enough
    - Full council deliberation runs
    - Result is cached for future similar questions

11. **Test the similarity threshold:**
    - "Green tea benefits" → Cache hit
    - "Tea benefits" → Might be a miss (too different)
    - "Health benefits of green tea for weight loss" → Might hit (similar enough)

### What to Notice

- Semantic caching understands MEANING, not just exact matches
- Similar questions return cached results instantly
- Different questions still get full council treatment
- You save money on repetitive queries

### Quick Win Achieved!

You've enabled intelligent caching that saves money on similar queries.

---

## Use Case 10: Debate Mode - AI vs AI

**Complexity: ★★★★☆ (Advanced)**

### Motivation

Instead of models answering independently, have them DEBATE each other. Each model states a position, critiques others, defends their position, and a judge determines the winner. This produces deeply considered answers.

### Time to Complete: 7 minutes

### Features Used
- Debate mode
- Multi-round deliberation
- Position statements
- Critiques and rebuttals
- Chairman judgment

### Step-by-Step Instructions

1. **Open the settings panel**

2. **Find "Debate Mode" toggle**
   - Turn it ON (orange theme)
   - Notice the "DB" badge appears
   - You can also enable/disable the "Include Rebuttal" sub-option

3. **Click "New Conversation"**

4. **Ask a debatable question:**
   ```
   Is it better to rent or buy a home in today's economy?
   ```

5. **Watch the debate unfold:**

   **Round 1 - Positions:**
   - Each model states their position
   - Notice they're labeled "Position A", "Position B", etc. (anonymous)
   - Some might favor renting, others buying

   **Round 2 - Critiques:**
   - Each model critiques another's position
   - You'll see "A critiques B", "B critiques C", etc.
   - Watch them identify weaknesses in each other's arguments

   **Round 3 - Rebuttals (if enabled):**
   - Models defend their positions
   - They address the criticisms they received
   - Some might acknowledge valid points

   **Final - Judgment:**
   - Chairman evaluates the entire debate
   - Determines which position was strongest
   - Synthesizes the best answer incorporating all insights

6. **Try a more controversial topic:**
   ```
   Should AI-generated art be considered "real" art?
   ```

7. **Watch models with different "opinions" clash:**
   - See how they argue their positions
   - Notice the quality of critiques
   - Observe how rebuttals strengthen or weaken arguments

8. **Compare debate vs. normal mode:**
   - Turn OFF debate mode
   - Ask the same question
   - Notice the difference in depth and consideration

### What to Notice

- Debates produce more thoroughly considered answers
- Critiques expose weaknesses in reasoning
- Rebuttals force models to strengthen arguments
- The final synthesis benefits from the debate's depth

### Quick Win Achieved!

You've experienced AI-vs-AI debate - the most thorough way to explore complex topics.

---

## Bonus Use Cases (For the Curious)

Here are additional features to explore once you're comfortable:

### Bonus A: Organize with Tags
- Click "Tags" in the conversation header
- Add tags like "coding", "research", "creative"
- Filter conversations by tag in the sidebar

### Bonus B: Configure Your Model Council
- Click "Configure Models" in settings bar
- Add/remove council members
- Choose your chairman
- Save your custom configuration

### Bonus C: Weighted Consensus
- Enable "Weighted Consensus" in settings
- Models that historically perform better have more voting power
- Requires some queries to build history (see Performance Dashboard)

### Bonus D: Early Consensus Exit
- Enable "Early Consensus" in settings
- If models strongly agree, skip the chairman synthesis
- Saves time and money when there's clear consensus

### Bonus E: Dynamic Model Routing
- Enable "Dynamic Routing" in settings
- Questions are classified (coding, creative, factual, etc.)
- Best models for each category are selected automatically

### Bonus F: Confidence-Gated Escalation
- Enable "Escalation" in settings
- Starts with cheaper Tier 1 models
- Only uses expensive Tier 2 models if confidence is low

### Bonus G: Iterative Refinement
- Enable "Refinement" in settings
- Council critiques the chairman's synthesis
- Chairman revises based on feedback
- Repeats until quality converges

### Bonus H: Adversarial Validation
- Enable "Adversary" in settings
- A "devil's advocate" model reviews the final answer
- Catches errors the council might have missed

### Bonus I: Sub-Question Decomposition
- Enable "Decomposition" in settings
- Complex questions are broken into parts
- Each part gets its own mini-council
- Results are merged into comprehensive answer

---

## What's Next?

Congratulations on completing the Quick Start guide! Here's where to go from here:

### Immediate Next Steps

1. **Read the full feature documentation:**
   - `docs/COMPREHENSIVE_FEATURE_GUIDE.md` - Detailed explanation of all 21 features

2. **Understand how it works:**
   - `docs/HOW_IT_WORKS.md` - Deep dive into the council process

3. **Learn the architecture:**
   - `docs/ARCHITECTURE.md` - Code structure and design decisions

### For Developers

4. **Extend the codebase:**
   - `docs/EXTENDING.md` - How to add new features
   - `CLAUDE.md` - Technical notes for AI-assisted development

5. **API Reference:**
   - `docs/API_REFERENCE.md` - All backend endpoints

### Troubleshooting

If you encounter issues:
- `docs/TROUBLESHOOTING.md` - Common problems and solutions
- Check that both terminals are still running
- Verify your API key has credits

### Join the Community

- Star the repo on GitHub
- Share your interesting council conversations
- Suggest new features or improvements

---

## Summary of Features by Use Case

| Use Case | Features | Complexity | Time |
|----------|----------|------------|------|
| 1. First Query | Basic council | ★☆☆☆☆ | 2 min |
| 2. Compare Perspectives | Peer review | ★☆☆☆☆ | 3 min |
| 3. Export Conversations | Export (MD/JSON) | ★☆☆☆☆ | 2 min |
| 4. Real-Time Streaming | Streaming, Process Monitor | ★★☆☆☆ | 3 min |
| 5. System Prompts | Custom behavior | ★★☆☆☆ | 5 min |
| 6. Performance Dashboard | Analytics | ★★☆☆☆ | 5 min |
| 7. Chain-of-Thought | Structured reasoning | ★★★☆☆ | 5 min |
| 8. Multi-Chairman | Ensemble synthesis | ★★★☆☆ | 5 min |
| 9. Response Caching | Semantic cache | ★★★☆☆ | 5 min |
| 10. Debate Mode | AI debates | ★★★★☆ | 7 min |

**Total Time to Complete All Use Cases: ~45 minutes**

---

Happy exploring! The LLM Council is now at your command.
