# Troubleshooting Guide

Common problems and their solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Startup Problems](#startup-problems)
3. [API Key Issues](#api-key-issues)
4. [Frontend Issues](#frontend-issues)
5. [Backend Issues](#backend-issues)
6. [Model/API Issues](#modelapi-issues)
7. [Data Issues](#data-issues)
8. [Performance Issues](#performance-issues)

---

## Installation Issues

### "uv: command not found"

**Problem**: uv package manager is not installed.

**Solution**:
```bash
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip:
pip install uv
```

After installation, restart your terminal.

---

### "npm: command not found"

**Problem**: Node.js is not installed.

**Solution**: Download and install from [nodejs.org](https://nodejs.org/).

Verify installation:
```bash
node --version
npm --version
```

---

### "Python version X.X is required"

**Problem**: Wrong Python version installed.

**Solution**: Install Python 3.10 or higher from [python.org](https://www.python.org/downloads/).

Check current version:
```bash
python --version
# or
python3 --version
```

---

### "npm install" fails with permission errors

**Problem**: Permission issues with npm.

**Solution**:
```bash
# Option 1: Use sudo (Linux/macOS)
sudo npm install

# Option 2: Fix npm permissions
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
# Add to ~/.bashrc or ~/.zshrc:
export PATH=~/.npm-global/bin:$PATH
```

---

## Startup Problems

### "ModuleNotFoundError: No module named 'backend'"

**Problem**: Running Python from wrong directory or without `-m` flag.

**Solution**:
```bash
# Make sure you're in the project root
cd llm-council-master

# Use -m flag to run as module
uv run python -m backend.main
```

---

### "Address already in use" (Port 8001)

**Problem**: Something else is using port 8001.

**Solution**:

Option 1: Find and kill the process
```bash
# On Linux/macOS:
lsof -i :8001
kill -9 <PID>

# On Windows:
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

Option 2: Use a different port (edit `backend/main.py`):
```python
uvicorn.run(app, host="0.0.0.0", port=8002)  # Change port
```

Remember to update `frontend/src/api.js`:
```javascript
const API_BASE = 'http://localhost:8002';
```

---

### "Address already in use" (Port 5173)

**Problem**: Vite dev server port is in use.

**Solution**:
```bash
# Specify different port
npm run dev -- --port 3000
```

Update CORS in `backend/main.py` to include the new port.

---

### Backend starts but frontend shows blank page

**Problem**: Frontend not built or not running.

**Solution**:
```bash
cd frontend
npm install  # If not done
npm run dev  # Start dev server
```

Make sure you see the Vite startup message with the URL.

---

## API Key Issues

### "Error querying model: 401 Unauthorized"

**Problem**: Invalid or missing API key.

**Solution**:

1. Check `.env` file exists in project root:
```bash
cat .env
# Should show: OPENROUTER_API_KEY=sk-or-v1-...
```

2. Verify key is valid at [openrouter.ai/keys](https://openrouter.ai/keys)

3. Check for typos or extra whitespace in the key

4. Make sure `.env` has no quotes around the value:
```bash
# Correct:
OPENROUTER_API_KEY=sk-or-v1-abc123

# Wrong:
OPENROUTER_API_KEY="sk-or-v1-abc123"
```

---

### "Error querying model: 402 Payment Required"

**Problem**: Insufficient OpenRouter credits.

**Solution**:
1. Go to [openrouter.ai/credits](https://openrouter.ai/credits)
2. Add credits to your account
3. Consider enabling auto top-up

---

### API key works in browser but not in code

**Problem**: Environment variable not loaded.

**Solution**:

1. Restart the backend after creating/modifying `.env`

2. Verify python-dotenv is installed:
```bash
uv run pip list | grep dotenv
```

3. Check the key is loaded:
```python
# Add to backend/config.py temporarily
print(f"API Key loaded: {OPENROUTER_API_KEY[:10]}...")
```

---

## Frontend Issues

### CORS Error: "Access-Control-Allow-Origin"

**Problem**: Frontend origin not allowed by backend.

**Solution**:

Check `backend/main.py` CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Add your frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Restart backend after changes.

---

### "Failed to fetch" errors

**Problem**: Backend is not running or not reachable.

**Solution**:

1. Verify backend is running:
```bash
curl http://localhost:8001/
# Should return: {"status":"ok","service":"LLM Council API"}
```

2. Check browser console for detailed error

3. Ensure frontend is using correct API URL in `frontend/src/api.js`

---

### Blank screen with no errors

**Problem**: React rendering error.

**Solution**:

1. Open browser DevTools (F12) → Console tab
2. Look for React errors
3. Common fix: Clear browser cache and reload
4. Try: `npm run build` to check for build errors

---

### Styles not loading / UI looks broken

**Problem**: CSS not imported or cached.

**Solution**:

1. Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)

2. Clear Vite cache:
```bash
cd frontend
rm -rf node_modules/.vite
npm run dev
```

3. Check CSS imports in components

---

## Backend Issues

### "ImportError: attempted relative import with no known parent package"

**Problem**: Running the file directly instead of as a module.

**Solution**:
```bash
# Wrong:
python backend/main.py
cd backend && python main.py

# Correct:
python -m backend.main
uv run python -m backend.main
```

---

### Responses are empty or null

**Problem**: Model queries are failing silently.

**Solution**:

1. Check terminal for error messages:
```
Error querying model openai/gpt-5.1: ...
```

2. Test API connectivity:
```bash
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}'
```

3. Try a different model (some may be temporarily unavailable)

---

### Timeout errors

**Problem**: Model taking too long to respond.

**Solution**:

1. Increase timeout in `backend/openrouter.py`:
```python
async def query_model(model, messages, timeout=180.0):  # Increase from 120
```

2. Use faster models (e.g., `google/gemini-2.5-flash`)

---

### Changes to code don't take effect

**Problem**: Server not restarting.

**Solution**:

1. Check uvicorn reload mode (should auto-reload):
```python
uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
```

2. Manually restart: Ctrl+C and run again

---

## Model/API Issues

### "Model not found" errors

**Problem**: Invalid model identifier.

**Solution**:

1. Check model exists at [openrouter.ai/models](https://openrouter.ai/models)

2. Use exact identifier format: `provider/model-name`

3. Check for typos in `backend/config.py`

---

### Rankings not being parsed correctly

**Problem**: Model not following the ranking format.

**How the parser works** (3 fallback levels):

1. **Primary**: Looks for `FINAL RANKING:` header, extracts numbered list (`1. Response A`)
2. **Secondary**: If no numbered list, extracts any `Response [A-Z]` patterns after the header
3. **Fallback**: If no header, extracts all `Response [A-Z]` patterns from entire text

**Debugging steps**:

1. Check raw ranking text in Stage 2 tabs
2. Look for "FINAL RANKING:" section
3. Check "Extracted Ranking" section below each evaluation - this shows what was parsed
4. If extracted ranking is empty or wrong, the model deviated from format

**Solutions**:

1. **If rankings are partially extracted**: The fallback is working but format was imperfect - usually acceptable

2. **If rankings are completely wrong**: Improve the prompt in `backend/council.py`:
```python
ranking_prompt = f"""...

CRITICAL: You MUST end your response with "FINAL RANKING:" followed by a numbered list.
Without this exact format, your evaluation cannot be processed.
..."""
```

3. **If one model consistently fails to rank**: Consider removing it from council or using a different model

**Common causes**:
- Model outputs ranking in a different format (bullets instead of numbers)
- Model adds extra commentary after the ranking
- Model uses different labels ("Answer A" instead of "Response A")

---

### One model always fails

**Problem**: Specific model has issues.

**Solution**:

1. Remove from council temporarily:
```python
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    # "problematic/model",  # Commented out
    "anthropic/claude-sonnet-4.5",
]
```

2. Check OpenRouter status for that provider

3. The system continues with remaining models (graceful degradation)

---

## Data Issues

### Conversations not saving

**Problem**: File system permission issues.

**Solution**:

1. Check data directory exists:
```bash
ls -la data/conversations/
```

2. Create manually if needed:
```bash
mkdir -p data/conversations
```

3. Check write permissions:
```bash
chmod 755 data/conversations
```

---

### Old conversations showing wrong format

**Problem**: Data format changed after updates.

**Solution**:

1. View the JSON file:
```bash
cat data/conversations/<id>.json
```

2. Delete corrupted conversations:
```bash
rm data/conversations/<id>.json
```

3. Or delete all and start fresh:
```bash
rm -rf data/conversations/*
```

---

### "Conversation not found" after restart

**Problem**: Looking for conversation that doesn't exist.

**Solution**:

1. Conversations are stored in `data/conversations/`
2. If directory was deleted, conversations are lost
3. Create a new conversation in the UI

---

## Performance Issues

### Queries taking too long (>2 minutes)

**Problem**: Sequential processing or slow models.

**Solution**:

1. Verify parallel execution is working (check `asyncio.gather` usage)

2. Use faster models:
```python
COUNCIL_MODELS = [
    "google/gemini-2.5-flash",     # Very fast
    "anthropic/claude-3-haiku",    # Fast
    "openai/gpt-4o-mini",          # Fast
]
```

3. Reduce number of council members

---

### High API costs

**Problem**: Using expensive models or too many models.

**Solution**:

1. Check usage at [openrouter.ai/activity](https://openrouter.ai/activity)

2. Use cheaper models:
```python
COUNCIL_MODELS = [
    "google/gemini-2.5-flash",           # Cheap
    "meta-llama/llama-3.1-70b-instruct", # Cheap
]
```

3. Reduce council size (2-3 models instead of 4+)

---

### Memory issues

**Problem**: Large conversations using too much memory.

**Solution**:

1. Each conversation is loaded entirely into memory
2. Start new conversations periodically
3. Delete old conversation files

---

## Still Having Issues?

### Debugging Steps

1. **Check both terminal windows** for error messages

2. **Check browser DevTools** (F12):
   - Console tab for JavaScript errors
   - Network tab for failed requests

3. **Test API directly**:
```bash
# Health check
curl http://localhost:8001/

# List conversations
curl http://localhost:8001/api/conversations
```

4. **Verify environment**:
```bash
python --version  # Should be 3.10+
node --version    # Should be 18+
cat .env          # Should show API key
```

5. **Fresh start**:
```bash
# Kill all processes
# Delete node_modules and reinstall
cd frontend && rm -rf node_modules && npm install

# Delete Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Reinstall Python deps
uv sync

# Start fresh
```

### Getting More Help

1. Search for the error message online
2. Check OpenRouter documentation
3. Review FastAPI or React documentation for framework-specific issues
