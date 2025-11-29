# Glossary

Definitions of technical terms used in LLM Council and this documentation.

---

## A

### API (Application Programming Interface)
A set of rules and protocols that allows different software applications to communicate. In this project, the backend exposes a REST API that the frontend calls to send messages and retrieve conversations.

### API Key
A secret token that identifies and authenticates your requests to a service. Your OpenRouter API key allows you to access various LLM providers through their unified API.

### Async/Await
A Python and JavaScript pattern for handling asynchronous operations. `async` marks a function as asynchronous, and `await` pauses execution until an operation completes without blocking other code.

```python
async def fetch_data():
    result = await some_api_call()  # Waits here, but doesn't block other tasks
    return result
```

### Asynchronous
Operations that don't block program execution while waiting. Multiple async operations can run concurrently, making programs faster when dealing with I/O (network requests, file operations).

---

## B

### Backend
The server-side part of an application that handles data processing, business logic, and database operations. In this project, the Python/FastAPI code that runs on port 8001.

---

## C

### Chairman Model
In LLM Council, the model designated to synthesize the final answer from all individual responses and peer rankings. It receives the full context of Stage 1 and Stage 2 results.

### Component (React)
A reusable piece of UI in React. Components can be nested, receive data via props, and maintain their own state. Example: `Sidebar`, `ChatInterface`, `Stage1`.

### CORS (Cross-Origin Resource Sharing)
A security feature that controls which domains can access your API. The backend must explicitly allow the frontend's origin (localhost:5173) to make requests.

### Council Models
The set of LLMs that participate in the deliberation process. They each provide responses in Stage 1 and rank each other in Stage 2.

---

## D

### De-anonymization
The process of revealing the actual model names after anonymous peer review. In Stage 2, models see "Response A, B, C" but the UI shows actual model names for user readability.

### Dependency
A package or library that your code requires to function. Listed in `pyproject.toml` (Python) and `package.json` (JavaScript).

---

## E

### Endpoint
A specific URL path in an API that performs a particular function. Example: `POST /api/conversations/{id}/message` sends a message.

### Environment Variable
Configuration values stored outside your code, often for sensitive data like API keys. Accessed via `os.getenv()` in Python. Stored in `.env` files.

---

## F

### FastAPI
A modern Python web framework for building APIs. Known for automatic documentation, data validation, and async support. Used for the backend.

### Frontend
The client-side part of an application that runs in the user's browser. In this project, the React application that provides the user interface.

---

## G

### Graceful Degradation
Design philosophy where a system continues to function (with reduced capability) when some components fail. If one model fails to respond, the council continues with the others.

---

## H

### Hook (React)
Functions that let you "hook into" React features like state and lifecycle. Common hooks: `useState` (state management), `useEffect` (side effects), `useRef` (DOM references).

### HTTP (Hypertext Transfer Protocol)
The protocol used for communication between web browsers/clients and servers. Common methods: GET (retrieve), POST (create), PUT (update), DELETE (remove).

### httpx
A Python HTTP client library that supports async operations. Used in this project to make API calls to OpenRouter.

---

## J

### JSON (JavaScript Object Notation)
A lightweight data format used for storing and transmitting data. Human-readable, widely supported. Example:
```json
{
  "name": "John",
  "age": 30,
  "active": true
}
```

### JSX
A syntax extension for JavaScript that looks like HTML. Used in React to describe UI structure. Gets compiled to JavaScript function calls.

```jsx
const element = <h1>Hello, {name}!</h1>;
```

---

## L

### LLM (Large Language Model)
AI models trained on vast amounts of text that can generate human-like text, answer questions, and perform various language tasks. Examples: GPT-4, Claude, Gemini, Llama.

---

## M

### Markdown
A lightweight markup language for formatting text. Uses symbols like `#` for headings, `*` for emphasis, and ``` for code blocks. Rendered by `react-markdown` in the frontend.

### Middleware
Code that runs between receiving a request and sending a response. CORS middleware in FastAPI handles cross-origin request headers.

### Model Identifier
The string that identifies a specific LLM on OpenRouter. Format: `provider/model-name`. Example: `openai/gpt-5.1`, `anthropic/claude-sonnet-4.5`.

---

## N

### npm (Node Package Manager)
The default package manager for Node.js. Used to install JavaScript dependencies and run scripts defined in `package.json`.

---

## O

### OpenRouter
A service that provides unified access to multiple LLM providers through a single API. Instead of managing separate API keys for OpenAI, Anthropic, Google, etc., you use one OpenRouter key.

### Optimistic Update
A UI pattern where changes are shown immediately before server confirmation. If the server request fails, the changes are rolled back. Provides faster-feeling UX.

---

## P

### Parallel Execution
Running multiple operations simultaneously rather than one after another. In this project, all council models are queried at the same time using `asyncio.gather()`.

### Parse/Parsing
Analyzing text or data to extract structured information. The backend parses model responses to extract rankings from the "FINAL RANKING:" section.

### Peer Review
In LLM Council, the process where each model evaluates and ranks all responses. Anonymization ensures unbiased evaluation.

### Props (Properties)
In React, the way data is passed from parent components to child components. Props are read-only.

```jsx
<Sidebar conversations={convList} onSelect={handleSelect} />
```

### Pydantic
A Python library for data validation using type hints. FastAPI uses Pydantic models to validate request/response data.

---

## R

### React
A JavaScript library for building user interfaces. Uses a component-based architecture and virtual DOM for efficient updates.

### Regex (Regular Expression)
A pattern-matching language for searching and manipulating text. Used in this project to extract rankings from model responses.

```python
re.findall(r'Response [A-Z]', text)  # Finds "Response A", "Response B", etc.
```

### REST (Representational State Transfer)
An architectural style for APIs that uses HTTP methods (GET, POST, PUT, DELETE) and URLs to perform operations on resources.

### Response
In HTTP: the data sent back from a server after a request.
In LLM Council: the text generated by a model in response to a query.

---

## S

### SSE (Server-Sent Events)
A technology for servers to push updates to clients over HTTP. Used for streaming stage completion events to the frontend.

Format: `data: {"type": "event_name", "data": {...}}\n\n`

### Stage
In LLM Council, one of the three phases of the deliberation process:
- Stage 1: Individual responses
- Stage 2: Peer rankings
- Stage 3: Chairman synthesis

### State
In React, data that can change over time and affects rendering. When state changes, the component re-renders.

```javascript
const [count, setCount] = useState(0);  // count is state
```

### Streaming
Sending data incrementally as it becomes available, rather than waiting for everything to complete. Provides real-time updates to users.

---

## T

### Token
In LLM context: the basic unit of text that models process. Roughly 4 characters or 0.75 words. API costs are based on tokens.

In authentication: a string that proves identity (like a session token or JWT).

---

## U

### UUID (Universally Unique Identifier)
A 128-bit identifier that's practically guaranteed to be unique. Used for conversation IDs. Example: `550e8400-e29b-41d4-a716-446655440000`

### uv
A fast Python package manager. Alternative to pip. Used in this project for dependency management. Commands: `uv sync` (install deps), `uv run` (run commands).

---

## V

### Vite
A fast build tool and development server for JavaScript projects. Used to run the React frontend. Features: hot module replacement, fast builds.

### Virtual Environment
An isolated Python environment with its own packages. Prevents conflicts between projects. Created by uv in `.venv/` directory.

---

## W

### WebSocket
A protocol for two-way communication between client and server. Unlike HTTP (request-response), WebSockets maintain an open connection. Not used in this project (SSE is used instead).

---

## Quick Reference

| Term | Simple Definition |
|------|-------------------|
| API | How programs talk to each other |
| Async | Non-blocking operations |
| Backend | Server code (Python) |
| Component | Reusable UI piece (React) |
| CORS | Security for cross-site requests |
| Endpoint | API URL for specific action |
| FastAPI | Python web framework |
| Frontend | Browser code (React) |
| Hook | React feature function |
| JSON | Data format |
| JSX | HTML-like React syntax |
| LLM | AI language model |
| Props | Data passed to components |
| REST | API design style |
| SSE | Server push technology |
| State | Changeable data in React |
| Token | Text unit / Auth credential |
| UUID | Unique ID |
