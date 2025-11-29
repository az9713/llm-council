# Frontend Guide

This guide explains every frontend file and component in detail. It's designed for developers new to React and modern JavaScript.

## Table of Contents

1. [Overview](#overview)
2. [Project Setup](#project-setup)
3. [Entry Point (main.jsx)](#entry-point-mainjsx)
4. [API Client (api.js)](#api-client-apijs)
5. [Export Utility (utils/export.js)](#export-utility-utilsexportjs)
6. [Main App Component (App.jsx)](#main-app-component-appjsx)
7. [Sidebar Component](#sidebar-component)
8. [ChatInterface Component](#chatinterface-component)
9. [Stage Components](#stage-components)
10. [Styling Overview](#styling-overview)
11. [React Concepts Explained](#react-concepts-explained)

---

## Overview

The frontend is a React application using:
- **React 19**: UI library for building components
- **Vite**: Fast build tool and development server
- **react-markdown**: Renders markdown content as HTML

File structure:
```
frontend/src/
├── main.jsx           # Entry point
├── App.jsx            # Main component
├── App.css            # App styles
├── api.js             # Backend API client
├── index.css          # Global styles
├── utils/
│   └── export.js      # Conversation export utilities
└── components/
    ├── Sidebar.jsx    # Conversation list + tag filter
    ├── Sidebar.css
    ├── ChatInterface.jsx  # Main chat area + export/tag buttons
    ├── ChatInterface.css
    ├── TagEditor.jsx  # Tag management component
    ├── TagEditor.css
    ├── ConfigPanel.jsx  # Model configuration panel
    ├── ConfigPanel.css
    ├── CostDisplay.jsx  # Cost breakdown display (NEW)
    ├── CostDisplay.css
    ├── Stage1.jsx     # Individual responses
    ├── Stage1.css
    ├── Stage2.jsx     # Peer rankings
    ├── Stage2.css
    ├── Stage3.jsx     # Final answer
    └── Stage3.css
```

---

## Project Setup

### Configuration Files

**package.json** - Defines dependencies and scripts:
```json
{
  "scripts": {
    "dev": "vite",        // Start development server
    "build": "vite build", // Build for production
    "preview": "vite preview" // Preview production build
  },
  "dependencies": {
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "react-markdown": "^10.1.0"
  }
}
```

**vite.config.js** - Vite configuration:
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
```

### Running the Frontend

```bash
cd frontend
npm install  # Install dependencies (first time only)
npm run dev  # Start development server
```

Opens at http://localhost:5173

---

## Entry Point (main.jsx)

**Location**: `frontend/src/main.jsx`

This is where the React application starts.

```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Find the HTML element with id="root" (in index.html)
// and render our React app inside it
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

### Explanation

- **createRoot**: React 18+ API for creating a root to render into
- **StrictMode**: Development helper that highlights potential problems
- **App**: The main application component

---

## API Client (api.js)

**Location**: `frontend/src/api.js`

Handles all communication with the backend.

```javascript
/**
 * API client for the LLM Council backend.
 */

// Backend URL - must match where the backend is running
const API_BASE = 'http://localhost:8001';

export const api = {
  /**
   * List all conversations, optionally filtered by tag.
   * @param {string} tag - Optional tag to filter by
   * @returns {Promise<Array>} List of conversation metadata
   */
  async listConversations(tag = null) {
    const url = tag
      ? `${API_BASE}/api/conversations?tag=${encodeURIComponent(tag)}`
      : `${API_BASE}/api/conversations`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to list conversations');
    }
    return response.json();
  },

  /**
   * Create a new conversation.
   * @returns {Promise<Object>} The new conversation
   */
  async createConversation() {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),  // Empty body, but still JSON
    });
    if (!response.ok) {
      throw new Error('Failed to create conversation');
    }
    return response.json();
  },

  /**
   * Get a specific conversation.
   * @param {string} conversationId - The conversation ID
   * @returns {Promise<Object>} The full conversation
   */
  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`
    );
    if (!response.ok) {
      throw new Error('Failed to get conversation');
    }
    return response.json();
  },

  /**
   * Send a message (non-streaming).
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {string} systemPrompt - Optional system prompt
   * @returns {Promise<Object>} The response with all stages
   */
  async sendMessage(conversationId, content, systemPrompt = null) {
    const body = { content };
    if (systemPrompt) {
      body.system_prompt = systemPrompt;
    }

    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    return response.json();
  },

  /**
   * Send a message and receive streaming updates.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback for each event: (eventType, data) => void
   * @param {string} systemPrompt - Optional system prompt
   * @returns {Promise<void>}
   */
  async sendMessageStream(conversationId, content, onEvent, systemPrompt = null) {
    const body = { content };
    if (systemPrompt) {
      body.system_prompt = systemPrompt;
    }

    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    // Get a reader for the response body stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    // Read the stream until it's done
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Decode the bytes to a string
      const chunk = decoder.decode(value);

      // SSE format: each event is "data: {...}\n\n"
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          // Extract the JSON part after "data: "
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            // Call the callback with event type and full event
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }
  },

  /**
   * Update tags for a conversation.
   * @param {string} conversationId - The conversation ID
   * @param {Array<string>} tags - Array of tags to set
   * @returns {Promise<Object>} Response with status and tags
   */
  async updateTags(conversationId, tags) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/tags`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to update tags');
    }
    return response.json();
  },

  /**
   * Get all unique tags across all conversations.
   * @returns {Promise<Object>} Object with tags array
   */
  async getAllTags() {
    const response = await fetch(`${API_BASE}/api/tags`);
    if (!response.ok) {
      throw new Error('Failed to get tags');
    }
    return response.json();
  },

  // ============ MODEL CONFIGURATION API ============

  /**
   * Get current model configuration.
   * @returns {Promise<{council_models: string[], chairman_model: string}>}
   */
  async getConfig() {
    const response = await fetch(`${API_BASE}/api/config`);
    if (!response.ok) {
      throw new Error('Failed to get config');
    }
    return response.json();
  },

  /**
   * Update model configuration.
   * @param {string[]} councilModels - Array of model identifiers
   * @param {string} chairmanModel - Chairman model identifier
   * @returns {Promise<{council_models: string[], chairman_model: string}>}
   */
  async updateConfig(councilModels, chairmanModel) {
    const response = await fetch(`${API_BASE}/api/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        council_models: councilModels,
        chairman_model: chairmanModel,
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update config');
    }
    return response.json();
  },

  /**
   * Reset configuration to defaults.
   * @returns {Promise<{council_models: string[], chairman_model: string}>}
   */
  async resetConfig() {
    const response = await fetch(`${API_BASE}/api/config/reset`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to reset config');
    }
    return response.json();
  },

  /**
   * Get list of suggested models for dropdown.
   * @returns {Promise<{models: string[]}>}
   */
  async getAvailableModels() {
    const response = await fetch(`${API_BASE}/api/config/models`);
    if (!response.ok) {
      throw new Error('Failed to get available models');
    }
    return response.json();
  },
};
```

### Key Concepts

**fetch()**: Built-in browser API for making HTTP requests. Returns a Promise.

**async/await**: JavaScript syntax for handling asynchronous operations. `await` pauses until the Promise resolves.

**response.body.getReader()**: For streaming responses, we can read the body incrementally instead of waiting for the whole response.

**Server-Sent Events (SSE)**: Format where each event is `data: {json}\n\n`. We parse these as they arrive.

---

## Export Utility (utils/export.js)

**Location**: `frontend/src/utils/export.js`

Provides functions to export conversations to different file formats.

```javascript
/**
 * Export utilities for LLM Council conversations.
 */

/**
 * Export conversation to Markdown format.
 * @param {Object} conversation - The conversation object
 */
export function exportToMarkdown(conversation) {
  const markdown = generateMarkdown(conversation);
  const filename = sanitizeFilename(conversation.title || 'conversation') + '.md';
  downloadFile(markdown, filename, 'text/markdown');
}

/**
 * Export conversation to JSON format.
 * @param {Object} conversation - The conversation object
 */
export function exportToJSON(conversation) {
  const json = JSON.stringify(conversation, null, 2);
  const filename = sanitizeFilename(conversation.title || 'conversation') + '.json';
  downloadFile(json, filename, 'application/json');
}

/**
 * Generate Markdown content from a conversation.
 * Formats all 3 stages with model names and includes parsed rankings.
 */
function generateMarkdown(conversation) {
  // Builds markdown with:
  // - Title and export timestamp
  // - User questions
  // - Stage 1: Individual responses with model names
  // - Stage 2: Peer rankings with extracted rankings
  // - Stage 3: Final synthesis with chairman model
  // ...
}

/**
 * Sanitize a string for use as a filename.
 * Removes invalid characters and limits length.
 */
function sanitizeFilename(name) {
  return name
    .replace(/[<>:"/\\|?*]/g, '')  // Remove invalid characters
    .replace(/\s+/g, '_')           // Replace spaces with underscores
    .substring(0, 50);               // Limit length
}

/**
 * Trigger a file download in the browser using Blob API.
 */
function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);  // Clean up
}
```

### Key Concepts

**Blob API**: Creates a file-like object from content. `URL.createObjectURL()` creates a temporary URL to download it.

**Client-side generation**: No server needed - everything happens in the browser, which is faster and works offline.

**Filename sanitization**: Important for cross-platform compatibility. Windows/Mac/Linux have different invalid characters.

---

## Main App Component (App.jsx)

**Location**: `frontend/src/App.jsx`

The root component that orchestrates the entire application.

```jsx
import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import './App.css';

function App() {
  // STATE MANAGEMENT
  // useState creates a piece of state and a function to update it

  // List of all conversations (metadata only)
  const [conversations, setConversations] = useState([]);

  // Currently selected conversation ID
  const [currentConversationId, setCurrentConversationId] = useState(null);

  // Full data for the current conversation
  const [currentConversation, setCurrentConversation] = useState(null);

  // Loading state during message processing
  const [isLoading, setIsLoading] = useState(false);

  // System prompt state (persisted to localStorage)
  const [systemPrompt, setSystemPrompt] = useState(
    () => localStorage.getItem('systemPrompt') || ''
  );

  // Settings panel visibility toggle
  const [showSettings, setShowSettings] = useState(false);

  // Tag-related state
  const [allTags, setAllTags] = useState([]);        // All unique tags across conversations
  const [selectedTag, setSelectedTag] = useState(null); // Currently selected tag filter

  // EFFECTS
  // useEffect runs code in response to component lifecycle events

  // Load conversations and tags when component first mounts
  useEffect(() => {
    loadConversations();
    loadAllTags();
  }, []);

  // Reload conversations when tag filter changes
  useEffect(() => {
    loadConversations(selectedTag);
  }, [selectedTag]);

  // Load conversation details whenever the selected ID changes
  useEffect(() => {
    if (currentConversationId) {
      loadConversation(currentConversationId);
    }
  }, [currentConversationId]);  // Dependency: run when this value changes

  // ASYNC FUNCTIONS FOR DATA LOADING

  const loadConversations = async (tag = null) => {
    try {
      const convs = await api.listConversations(tag);
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadAllTags = async () => {
    try {
      const result = await api.getAllTags();
      setAllTags(result.tags);
    } catch (error) {
      console.error('Failed to load tags:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  // EVENT HANDLERS

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      // Clear tag filter when creating new conversation
      setSelectedTag(null);
      // Add to list (at beginning) and select it
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, tags: [], message_count: 0 },
        ...conversations,  // Spread existing conversations after
      ]);
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
  };

  // Handle tag changes for the current conversation
  const handleTagsChange = async (tags) => {
    if (!currentConversationId) return;

    try {
      await api.updateTags(currentConversationId, tags);
      // Update current conversation state
      setCurrentConversation((prev) => ({ ...prev, tags }));
      // Reload conversations list to reflect tag changes
      loadConversations(selectedTag);
      // Reload all tags in case new tags were added
      loadAllTags();
    } catch (error) {
      console.error('Failed to update tags:', error);
    }
  };

  // Handle tag filter dropdown changes
  const handleTagFilterChange = (tag) => {
    setSelectedTag(tag);
  };

  // Handle system prompt changes (persisted to localStorage)
  const handleSystemPromptChange = (value) => {
    setSystemPrompt(value);
    localStorage.setItem('systemPrompt', value);
  };

  const handleSendMessage = async (content) => {
    if (!currentConversationId) return;

    setIsLoading(true);
    try {
      // OPTIMISTIC UPDATE: Show user message immediately
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Create placeholder for assistant response
      // Will be updated as streaming events arrive
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        loading: {
          stage1: false,
          stage2: false,
          stage3: false,
        },
      };

      // Add placeholder to messages
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Send with streaming, handling each event type
      // Pass systemPrompt as the 4th argument
      await api.sendMessageStream(
        currentConversationId,
        content,
        (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            // Mark Stage 1 as loading
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage1 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage1_complete':
            // Store Stage 1 data, clear loading
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage1 = event.data;
              lastMsg.loading.stage1 = false;
              return { ...prev, messages };
            });
            break;

          case 'stage2_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage2 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage2_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage2 = event.data;
              lastMsg.metadata = event.metadata;  // Includes label_to_model
              lastMsg.loading.stage2 = false;
              return { ...prev, messages };
            });
            break;

          case 'stage3_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage3 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage3_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage3 = event.data;
              lastMsg.loading.stage3 = false;
              return { ...prev, messages };
            });
            break;

          case 'title_complete':
            // Reload conversations to show new title
            loadConversations();
            break;

          case 'complete':
            // Everything done
            loadConversations();
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            break;

          default:
            console.log('Unknown event type:', eventType);
        }
      },
        systemPrompt || null  // Pass system prompt to backend
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),  // Remove last 2 messages
      }));
      setIsLoading(false);
    }
  };

  // RENDER
  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        allTags={allTags}
        selectedTag={selectedTag}
        onTagFilterChange={handleTagFilterChange}
      />
      <div className="main-content">
        {/* Settings Panel - collapsible system prompt configuration */}
        <div className="settings-bar">
          <button
            className="settings-toggle"
            onClick={() => setShowSettings(!showSettings)}
          >
            {showSettings ? '▼ Hide Settings' : '▶ Settings'}
            {/* Blue dot indicator when system prompt is active */}
            {systemPrompt && <span className="settings-active-indicator">●</span>}
          </button>
          {showSettings && (
            <div className="settings-panel">
              <label htmlFor="system-prompt">System Prompt</label>
              <textarea
                id="system-prompt"
                value={systemPrompt}
                onChange={(e) => handleSystemPromptChange(e.target.value)}
                placeholder="Enter a system prompt..."
                rows={3}
              />
              {systemPrompt && (
                <button onClick={() => handleSystemPromptChange('')}>
                  Clear
                </button>
              )}
            </div>
          )}
        </div>
        <ChatInterface
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          onTagsChange={handleTagsChange}
        />
      </div>
    </div>
  );
}

export default App;
```

### Key Concepts Explained

**useState**: Creates reactive state. When state changes, the component re-renders.
```jsx
const [value, setValue] = useState(initialValue);
// value - current value
// setValue - function to update it
// initialValue - starting value
```

**useEffect**: Runs side effects (API calls, subscriptions, etc.)
```jsx
useEffect(() => {
  // Code to run
}, [dependencies]);
// Empty [] = run once on mount
// [value] = run when value changes
```

**Optimistic Updates**: Show changes immediately before server confirms. Better UX but requires error handling to rollback.

**Spread Operator**: `...` spreads array/object contents
```jsx
[newItem, ...existingArray]  // New item first
{...prev, newKey: newValue}   // Copy object with changes
```

---

## Sidebar Component

**Location**: `frontend/src/components/Sidebar.jsx`

Displays the conversation list, tag filter dropdown, and new conversation button.

```jsx
import './Sidebar.css';

export default function Sidebar({
  conversations,           // Array of conversation metadata
  currentConversationId,   // Currently selected ID
  onSelectConversation,    // Callback when conversation clicked
  onNewConversation,       // Callback when new conversation clicked
  allTags,                 // Array of all unique tags
  selectedTag,             // Currently selected tag filter (or null)
  onTagFilterChange,       // Callback when tag filter changes
}) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>LLM Council</h1>
        <button className="new-conversation-btn" onClick={onNewConversation}>
          + New Conversation
        </button>
      </div>

      {/* Tag Filter Dropdown */}
      {allTags && allTags.length > 0 && (
        <div className="tag-filter">
          <select
            className="tag-filter-select"
            value={selectedTag || ''}
            onChange={(e) => onTagFilterChange(e.target.value || null)}
          >
            <option value="">All conversations</option>
            {allTags.map((tag) => (
              <option key={tag} value={tag}>
                #{tag}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="conversation-list">
        {conversations.map((conv) => (
          <div
            key={conv.id}  // React needs unique key for list items
            className={`conversation-item ${
              conv.id === currentConversationId ? 'active' : ''
            }`}
            onClick={() => onSelectConversation(conv.id)}
          >
            <div className="conversation-title">
              {conv.title || 'New Conversation'}
            </div>
            <div className="conversation-meta">
              {conv.message_count} messages
            </div>
            {/* Display conversation tags */}
            {conv.tags && conv.tags.length > 0 && (
              <div className="conversation-tags">
                {conv.tags.map((tag) => (
                  <span key={tag} className="conversation-tag">#{tag}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Props

Props are inputs to a component, passed from parent to child.

| Prop | Type | Description |
|------|------|-------------|
| `conversations` | Array | List of conversation metadata (includes tags) |
| `currentConversationId` | String | ID of selected conversation |
| `onSelectConversation` | Function | Called with ID when clicked |
| `onNewConversation` | Function | Called when new button clicked |
| `allTags` | Array | All unique tags for filter dropdown |
| `selectedTag` | String/null | Currently active tag filter |
| `onTagFilterChange` | Function | Called with tag (or null) when filter changes |

---

## TagEditor Component

**Location**: `frontend/src/components/TagEditor.jsx`

Provides an interface for adding, removing, and selecting tags with suggested options.

```jsx
import { useState } from 'react';
import './TagEditor.css';

// Predefined suggested tags for quick selection
const SUGGESTED_TAGS = [
  'coding', 'writing', 'analysis', 'research',
  'creative', 'technical', 'business', 'learning',
];

export default function TagEditor({ tags, onTagsChange }) {
  const [inputValue, setInputValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Add a tag (normalize to lowercase, avoid duplicates)
  const addTag = (tag) => {
    const normalizedTag = tag.trim().toLowerCase();
    if (normalizedTag && !tags.includes(normalizedTag)) {
      onTagsChange([...tags, normalizedTag]);
    }
    setInputValue('');
    setShowSuggestions(false);
  };

  // Remove a tag from the list
  const removeTag = (tagToRemove) => {
    onTagsChange(tags.filter((t) => t !== tagToRemove));
  };

  // Handle Enter key to add custom tag
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (inputValue.trim()) {
        addTag(inputValue);
      }
    }
  };

  // Filter suggested tags to only show unused ones
  const availableSuggestions = SUGGESTED_TAGS.filter(
    (tag) => !tags.includes(tag)
  );

  return (
    <div className="tag-editor">
      {/* Display current tags */}
      <div className="tags-list">
        {tags.map((tag) => (
          <span key={tag} className="tag">
            #{tag}
            <button className="tag-remove" onClick={() => removeTag(tag)}>×</button>
          </span>
        ))}
      </div>

      {/* Input for custom tags */}
      <input
        type="text"
        className="tag-input"
        placeholder="Add tag..."
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => setShowSuggestions(true)}
        onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
      />

      {/* Suggested tags dropdown */}
      {showSuggestions && availableSuggestions.length > 0 && (
        <div className="tag-suggestions">
          {availableSuggestions.map((tag) => (
            <button
              key={tag}
              className="tag-suggestion"
              onClick={() => addTag(tag)}
            >
              #{tag}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
```

### Props

| Prop | Type | Description |
|------|------|-------------|
| `tags` | Array | Current array of tag strings |
| `onTagsChange` | Function | Called with new tags array when changed |

### Key Features

- **Suggested tags**: Predefined list for quick selection (coding, writing, etc.)
- **Custom tags**: Type and press Enter to add any tag
- **Normalization**: Tags are converted to lowercase to ensure consistency
- **Duplicate prevention**: Cannot add the same tag twice
- **Easy removal**: Click × to remove tags

---

## ConfigPanel Component

**Location**: `frontend/src/components/ConfigPanel.jsx`

A modal panel for managing model configuration without editing code.

```jsx
import { useState, useEffect } from 'react';
import { api } from '../api';
import './ConfigPanel.css';

export default function ConfigPanel({ onClose }) {
  // Configuration state
  const [councilModels, setCouncilModels] = useState([]);
  const [chairmanModel, setChairmanModel] = useState('');

  // Available models for autocomplete suggestions
  const [availableModels, setAvailableModels] = useState([]);

  // UI state
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Input for adding new models
  const [newModelInput, setNewModelInput] = useState('');
  const [showModelDropdown, setShowModelDropdown] = useState(false);

  // Load config and available models on mount
  useEffect(() => {
    loadConfig();
    loadAvailableModels();
  }, []);

  const loadConfig = async () => {
    try {
      setIsLoading(true);
      const config = await api.getConfig();
      setCouncilModels(config.council_models);
      setChairmanModel(config.chairman_model);
      setError(null);
    } catch (err) {
      setError('Failed to load configuration');
    } finally {
      setIsLoading(false);
    }
  };

  const loadAvailableModels = async () => {
    try {
      const result = await api.getAvailableModels();
      setAvailableModels(result.models);
    } catch (err) {
      console.error('Failed to load available models:', err);
    }
  };

  const handleSave = async () => {
    // Validate: at least 2 models required
    if (councilModels.length < 2) {
      setError('At least 2 council models are required');
      return;
    }
    if (!chairmanModel.trim()) {
      setError('Chairman model is required');
      return;
    }

    try {
      setIsSaving(true);
      setError(null);
      await api.updateConfig(councilModels, chairmanModel);
      setSuccessMessage('Configuration saved successfully!');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err.message || 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Reset configuration to defaults?')) return;

    try {
      setIsSaving(true);
      const config = await api.resetConfig();
      setCouncilModels(config.council_models);
      setChairmanModel(config.chairman_model);
      setSuccessMessage('Configuration reset to defaults');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError('Failed to reset configuration');
    } finally {
      setIsSaving(false);
    }
  };

  // Add model to council (prevent duplicates)
  const addModel = (model) => {
    const trimmed = model.trim();
    if (!trimmed || councilModels.includes(trimmed)) return;
    setCouncilModels([...councilModels, trimmed]);
    setNewModelInput('');
    setShowModelDropdown(false);
  };

  // Remove model from council (minimum 2 required)
  const removeModel = (index) => {
    if (councilModels.length <= 2) {
      setError('At least 2 council models are required');
      setTimeout(() => setError(null), 2000);
      return;
    }
    const removed = councilModels[index];
    const newModels = councilModels.filter((_, i) => i !== index);
    setCouncilModels(newModels);

    // Update chairman if it was removed
    if (removed === chairmanModel) {
      setChairmanModel(newModels[0] || '');
    }
  };

  // Reorder models (move up/down)
  const moveModel = (index, direction) => {
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= councilModels.length) return;

    const newModels = [...councilModels];
    [newModels[index], newModels[newIndex]] = [newModels[newIndex], newModels[index]];
    setCouncilModels(newModels);
  };

  // Filter suggestions for dropdown
  const filteredModels = availableModels.filter(
    (model) =>
      !councilModels.includes(model) &&
      model.toLowerCase().includes(newModelInput.toLowerCase())
  );

  // ... render JSX with:
  // - Model list with reorder/remove buttons
  // - Add model input with autocomplete dropdown
  // - Quick-add buttons for popular models
  // - Chairman selection dropdown
  // - Save/Reset/Cancel buttons
}
```

### Props

| Prop | Type | Description |
|------|------|-------------|
| `onClose` | Function | Called when panel should be closed |

### Key Features

- **Add/Remove Models**: Manage council members dynamically
- **Reorder Models**: Move models up/down in the list
- **Chairman Selection**: Dropdown to pick chairman (can be council member or external)
- **Model Autocomplete**: Suggestions from backend while typing
- **Quick-Add Buttons**: One-click add for popular models
- **Validation**: Enforces minimum 2 council models
- **Reset to Defaults**: Restore original configuration
- **Error/Success Messages**: User feedback for actions

### Integration in App.jsx

The ConfigPanel is opened from a button in the settings bar:

```jsx
// State in App.jsx
const [showConfigPanel, setShowConfigPanel] = useState(false);

// Button in settings bar
<button
  className="config-models-btn"
  onClick={() => setShowConfigPanel(true)}
>
  Configure Models
</button>

// Render modal with overlay
{showConfigPanel && (
  <>
    <div className="config-overlay" onClick={() => setShowConfigPanel(false)} />
    <ConfigPanel onClose={() => setShowConfigPanel(false)} />
  </>
)}
```

---

## CostDisplay Component

**Location**: `frontend/src/components/CostDisplay.jsx`

Displays token usage and cost breakdown for council queries.

```jsx
import './CostDisplay.css';

/**
 * CostDisplay - Shows token usage and cost breakdown for a council query.
 *
 * @param {Object} costs - Cost summary from API (stage1, stage2, stage3, total)
 * @param {boolean} expanded - Show detailed breakdown (default: false)
 */
export default function CostDisplay({ costs, expanded = false }) {
  // Compact view: inline summary
  // Expanded view: full breakdown by stage

  const formatCost = (cost) => {
    if (cost < 0.0001) return '<$0.0001';
    if (cost < 0.01) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
  };

  // ...
}

/**
 * ModelCostBadge - Inline cost badge for individual model responses.
 */
export function ModelCostBadge({ usage, cost }) {
  // Shows tokens and cost in a small badge
}
```

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `costs` | Object | required | Cost summary with stage1, stage2, stage3, total |
| `expanded` | boolean | false | Show detailed breakdown vs compact view |

### Cost Object Structure

```javascript
{
  stage1: {
    total_cost: 0.0045,
    total_input_cost: 0.001,
    total_output_cost: 0.0035,
    total_tokens: 1500,
    total_prompt_tokens: 400,
    total_completion_tokens: 1100,
    model_count: 4,
  },
  stage2: { /* same structure */ },
  stage3: { /* same structure, model_count: 1 */ },
  total: { /* aggregated across all stages */ },
}
```

### Key Features

- **Compact View**: Inline summary showing total cost and tokens
- **Expanded View**: Full breakdown with per-stage costs and percentages
- **Cost Formatting**: Smart formatting (<$0.0001, $0.0012, $1.23)
- **Token Counts**: Shows prompt + completion tokens separately
- **Percentage Distribution**: Shows what portion each stage consumed
- **Green Color Scheme**: Indicates cost/value information

### Integration

CostDisplay is rendered in ChatInterface after Stage3:

```jsx
{msg.metadata?.costs && (
  <CostDisplay costs={msg.metadata.costs} expanded={true} />
)}
```

App.jsx handles the `costs_complete` streaming event:

```jsx
case 'costs_complete':
  setCurrentConversation((prev) => {
    const messages = [...prev.messages];
    const lastMsg = messages[messages.length - 1];
    lastMsg.metadata = { ...lastMsg.metadata, costs: event.data };
    return { ...prev, messages };
  });
  break;
```

---

## ChatInterface Component

**Location**: `frontend/src/components/ChatInterface.jsx`

The main chat area with messages, input, and export functionality.

```jsx
import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import TagEditor from './TagEditor';
import { exportToMarkdown, exportToJSON } from '../utils/export';
import './ChatInterface.css';

export default function ChatInterface({ conversation, onSendMessage, isLoading, onTagsChange }) {
  // Local state for the input field
  const [input, setInput] = useState('');

  // State for showing/hiding the tag editor panel
  const [showTagEditor, setShowTagEditor] = useState(false);

  // Ref to the messages container for scrolling
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages]);

  const handleSubmit = (e) => {
    e.preventDefault();  // Prevent form from reloading page
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');  // Clear input after sending
    }
  };

  const handleKeyDown = (e) => {
    // Enter sends message, Shift+Enter adds new line
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Show prompt if no conversation selected
  if (!conversation) {
    return (
      <div className="chat-interface">
        <div className="no-conversation">
          <p>Select a conversation or create a new one to get started</p>
        </div>
      </div>
    );
  }

  const hasMessages = conversation.messages.length > 0;

  return (
    <div className="chat-interface">
      {/* Header with title, tags, and action buttons */}
      {hasMessages && (
        <div className="chat-header">
          <div className="chat-header-left">
            <h2 className="chat-title">{conversation.title || 'Conversation'}</h2>
            {/* Display tags in header when not editing */}
            {conversation.tags && conversation.tags.length > 0 && !showTagEditor && (
              <div className="header-tags">
                {conversation.tags.map((tag) => (
                  <span key={tag} className="header-tag">#{tag}</span>
                ))}
              </div>
            )}
          </div>
          <div className="chat-header-actions">
            <button
              className={`action-btn ${showTagEditor ? 'active' : ''}`}
              onClick={() => setShowTagEditor(!showTagEditor)}
              title="Edit tags"
            >
              Tags
            </button>
            <button
              className="action-btn"
              onClick={() => exportToMarkdown(conversation)}
              title="Export to Markdown"
            >
              Export MD
            </button>
            <button
              className="action-btn"
              onClick={() => exportToJSON(conversation)}
              title="Export to JSON"
            >
              Export JSON
            </button>
          </div>
        </div>
      )}

      {/* Tag Editor Panel (collapsible) */}
      {hasMessages && showTagEditor && (
        <div className="tag-editor-container">
          <TagEditor
            tags={conversation.tags || []}
            onTagsChange={onTagsChange}
          />
        </div>
      )}

      <div className="messages">
        {conversation.messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            {message.role === 'user' ? (
              // User messages: simple markdown
              <div className="user-message markdown-content">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
            ) : (
              // Assistant messages: show all stages
              <div className="assistant-message">
                {/* Stage 1: Individual Responses */}
                {message.loading?.stage1 && (
                  <div className="loading-indicator">
                    <span className="spinner"></span>
                    Stage 1: Collecting responses...
                  </div>
                )}
                {message.stage1 && <Stage1 responses={message.stage1} />}

                {/* Stage 2: Peer Rankings */}
                {message.loading?.stage2 && (
                  <div className="loading-indicator">
                    <span className="spinner"></span>
                    Stage 2: Collecting rankings...
                  </div>
                )}
                {message.stage2 && (
                  <Stage2
                    rankings={message.stage2}
                    labelToModel={message.metadata?.label_to_model}
                    aggregateRankings={message.metadata?.aggregate_rankings}
                  />
                )}

                {/* Stage 3: Final Answer */}
                {message.loading?.stage3 && (
                  <div className="loading-indicator">
                    <span className="spinner"></span>
                    Stage 3: Synthesizing final answer...
                  </div>
                )}
                {message.stage3 && <Stage3 response={message.stage3} />}
              </div>
            )}
          </div>
        ))}
        {/* Invisible element at bottom for scrolling target */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form className="message-input" onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          rows={3}
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}
```

### Key Concepts

**useRef**: Creates a reference to a DOM element. Used here for scroll-to-bottom.

**Conditional Rendering**: `{condition && <Component />}` only renders if condition is true.

**Optional Chaining**: `conversation?.messages` safely accesses nested properties (returns undefined if conversation is null).

**Form Handling**: `onSubmit` handles form submission, `e.preventDefault()` stops page reload.

---

## Stage Components

### Stage1.jsx - Individual Responses

Displays individual model responses with support for reasoning models (o1, o3, etc.) that expose their thinking process.

```jsx
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage1.css';

/**
 * Stage1 - Displays individual model responses with optional reasoning details.
 *
 * Reasoning models (like OpenAI o1, o3) return a separate "reasoning_details"
 * field containing their chain-of-thought process. This component displays
 * that reasoning in a collapsible section above the final response.
 */
export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);
  const [showReasoning, setShowReasoning] = useState({});

  if (!responses || responses.length === 0) {
    return null;
  }

  // Toggle reasoning visibility for a specific model index
  const toggleReasoning = (index) => {
    setShowReasoning((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const currentResponse = responses[activeTab];
  const hasReasoning = currentResponse.reasoning_details != null;

  // Format reasoning details for display
  // Handles various formats: string, object with content, array of steps
  const formatReasoning = (reasoning) => {
    if (!reasoning) return '';
    if (typeof reasoning === 'string') return reasoning;
    if (typeof reasoning === 'object') {
      if (reasoning.content) return reasoning.content;
      if (Array.isArray(reasoning)) {
        return reasoning
          .map((item) => (typeof item === 'string' ? item : item.content || JSON.stringify(item)))
          .join('\n\n');
      }
      return JSON.stringify(reasoning, null, 2);
    }
    return String(reasoning);
  };

  return (
    <div className="stage stage1">
      <h3 className="stage-title">Stage 1: Individual Responses</h3>

      {/* Tab buttons - asterisk indicates reasoning model */}
      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''} ${resp.reasoning_details ? 'has-reasoning' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {resp.model.split('/')[1] || resp.model}
            {resp.reasoning_details && <span className="reasoning-indicator">*</span>}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">
          {currentResponse.model}
          {hasReasoning && <span className="reasoning-model-badge">Reasoning Model</span>}
        </div>

        {/* Collapsible Reasoning Section */}
        {hasReasoning && (
          <div className="reasoning-section">
            <button
              className={`reasoning-toggle ${showReasoning[activeTab] ? 'expanded' : ''}`}
              onClick={() => toggleReasoning(activeTab)}
            >
              <span className="reasoning-toggle-icon">
                {showReasoning[activeTab] ? '▼' : '▶'}
              </span>
              <span className="reasoning-toggle-text">
                {showReasoning[activeTab] ? 'Hide' : 'Show'} Thinking Process
              </span>
            </button>

            {showReasoning[activeTab] && (
              <div className="reasoning-content">
                <div className="reasoning-header">Chain of Thought</div>
                <div className="reasoning-text markdown-content">
                  <ReactMarkdown>
                    {formatReasoning(currentResponse.reasoning_details)}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Final Response */}
        <div className="response-section">
          {hasReasoning && <div className="response-label">Final Response</div>}
          <div className="response-text markdown-content">
            <ReactMarkdown>{currentResponse.response}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}
```

### Key Features

- **Reasoning Detection**: Checks for `reasoning_details` field in response data
- **Visual Indicators**: Asterisk (*) on tab, "Reasoning Model" badge next to model name
- **Collapsible Section**: Toggle button to show/hide thinking process
- **Amber/Gold Styling**: Distinct visual treatment for reasoning content
- **Format Handling**: Supports string, object with content, or array of steps
- **Graceful Fallback**: Works normally for non-reasoning models

### Stage2.jsx - Peer Rankings

```jsx
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage2.css';

// Helper function to replace anonymous labels with model names
function deAnonymizeText(text, labelToModel) {
  if (!labelToModel) return text;

  let result = text;
  // For each mapping, replace "Response X" with "**modelname**"
  Object.entries(labelToModel).forEach(([label, model]) => {
    const modelShortName = model.split('/')[1] || model;
    // Use regex with 'g' flag for global replacement
    result = result.replace(new RegExp(label, 'g'), `**${modelShortName}**`);
  });
  return result;
}

export default function Stage2({ rankings, labelToModel, aggregateRankings }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!rankings || rankings.length === 0) {
    return null;
  }

  return (
    <div className="stage stage2">
      <h3 className="stage-title">Stage 2: Peer Rankings</h3>

      <h4>Raw Evaluations</h4>
      <p className="stage-description">
        Each model evaluated all responses (anonymized as Response A, B, C, etc.)
        and provided rankings. Below, model names are shown in <strong>bold</strong>
        for readability, but the original evaluation used anonymous labels.
      </p>

      {/* Tab buttons for each model's evaluation */}
      <div className="tabs">
        {rankings.map((rank, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {rank.model.split('/')[1] || rank.model}
          </button>
        ))}
      </div>

      {/* Current model's evaluation */}
      <div className="tab-content">
        <div className="ranking-model">{rankings[activeTab].model}</div>
        <div className="ranking-content markdown-content">
          <ReactMarkdown>
            {/* De-anonymize for readability */}
            {deAnonymizeText(rankings[activeTab].ranking, labelToModel)}
          </ReactMarkdown>
        </div>

        {/* Show parsed/extracted ranking for transparency */}
        {rankings[activeTab].parsed_ranking &&
         rankings[activeTab].parsed_ranking.length > 0 && (
          <div className="parsed-ranking">
            <strong>Extracted Ranking:</strong>
            <ol>
              {rankings[activeTab].parsed_ranking.map((label, i) => (
                <li key={i}>
                  {/* Show actual model name if available */}
                  {labelToModel && labelToModel[label]
                    ? labelToModel[label].split('/')[1] || labelToModel[label]
                    : label}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>

      {/* Aggregate Rankings ("Street Cred") */}
      {aggregateRankings && aggregateRankings.length > 0 && (
        <div className="aggregate-rankings">
          <h4>Aggregate Rankings (Street Cred)</h4>
          <p className="stage-description">
            Combined results across all peer evaluations (lower score is better):
          </p>
          <div className="aggregate-list">
            {aggregateRankings.map((agg, index) => (
              <div key={index} className="aggregate-item">
                <span className="rank-position">#{index + 1}</span>
                <span className="rank-model">
                  {agg.model.split('/')[1] || agg.model}
                </span>
                <span className="rank-score">
                  Avg: {agg.average_rank.toFixed(2)}
                </span>
                <span className="rank-count">
                  ({agg.rankings_count} votes)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

### Stage3.jsx - Final Answer

```jsx
import ReactMarkdown from 'react-markdown';
import './Stage3.css';

export default function Stage3({ response }) {
  if (!response) {
    return null;
  }

  return (
    <div className="stage stage3">
      <h3 className="stage-title">Stage 3: Final Answer</h3>
      <div className="chairman-model">Chairman: {response.model}</div>
      <div className="final-response markdown-content">
        <ReactMarkdown>{response.response}</ReactMarkdown>
      </div>
    </div>
  );
}
```

---

## Styling Overview

### Global Styles (index.css)

```css
/* Reset and base styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color: #333;
}

/* Markdown content styling - used everywhere markdown is rendered */
.markdown-content {
  padding: 12px;  /* Prevents text from touching edges */
}

.markdown-content p {
  margin-bottom: 1em;
}

.markdown-content ul, .markdown-content ol {
  margin-left: 1.5em;
  margin-bottom: 1em;
}

.markdown-content code {
  background-color: #f4f4f4;
  padding: 0.2em 0.4em;
  border-radius: 3px;
  font-family: monospace;
}

.markdown-content pre {
  background-color: #f4f4f4;
  padding: 1em;
  border-radius: 4px;
  overflow-x: auto;
}
```

### Color Scheme

| Element | Color | CSS |
|---------|-------|-----|
| Primary (buttons, active) | Blue | `#4a90e2` |
| Background | White | `#ffffff` |
| Sidebar background | Light gray | `#f8f8f8` |
| User message background | Light blue | `#f0f7ff` |
| Stage 3 background | Light green | `#f0fff0` |
| Text | Dark gray | `#333` |

---

## React Concepts Explained

### Components

Components are reusable UI pieces. They can be:
- **Functional**: Simple functions that return JSX
- **Receiving props**: Data passed from parent
- **Maintaining state**: Internal data using `useState`

```jsx
// Functional component with props
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// Usage
<Greeting name="World" />
```

### JSX

JSX is JavaScript + HTML syntax:
```jsx
// JSX
const element = <div className="box">Hello</div>;

// Compiles to:
const element = React.createElement('div', {className: 'box'}, 'Hello');
```

Key differences from HTML:
- `className` instead of `class`
- `{}` for JavaScript expressions
- Self-closing tags required: `<img />`

### State Management

State is data that can change over time:
```jsx
const [count, setCount] = useState(0);

// Update state
setCount(5);           // Set to 5
setCount(c => c + 1);  // Increment (use function for updates based on current value)
```

### Effects

Effects handle side effects (things outside React's rendering):
```jsx
useEffect(() => {
  // Runs after render
  fetchData();

  // Optional cleanup (runs before next effect or unmount)
  return () => cleanup();
}, [dependency]);  // Only re-run if dependency changes
```

### Props vs State

| Props | State |
|-------|-------|
| Passed from parent | Owned by component |
| Read-only | Can be updated |
| Trigger re-render when changed | Trigger re-render when changed |

### Optimistic Updates Pattern

This codebase uses **optimistic updates** - showing changes immediately before the server confirms them. This makes the UI feel faster.

**How it works in App.jsx:**

```jsx
const handleSendMessage = async (content) => {
  // 1. IMMEDIATELY add user message to UI (before server responds)
  const userMessage = { role: 'user', content };
  setCurrentConversation((prev) => ({
    ...prev,
    messages: [...prev.messages, userMessage],
  }));

  // 2. Add placeholder for assistant response
  const assistantMessage = {
    role: 'assistant',
    stage1: null,  // Will be filled by SSE events
    stage2: null,
    stage3: null,
    loading: { stage1: false, stage2: false, stage3: false },
  };
  setCurrentConversation((prev) => ({
    ...prev,
    messages: [...prev.messages, assistantMessage],
  }));

  try {
    // 3. Send to server and update via streaming events
    await api.sendMessageStream(...);
  } catch (error) {
    // 4. ROLLBACK: Remove optimistic messages if server fails
    setCurrentConversation((prev) => ({
      ...prev,
      messages: prev.messages.slice(0, -2),  // Remove last 2 messages
    }));
  }
};
```

**Key points:**
- User sees their message instantly (no waiting for server)
- Placeholder message shows loading indicators
- SSE events progressively fill in the response
- If something fails, the optimistic messages are removed

---

## Next Steps

- **Understand the API**: [API Reference](./API_REFERENCE.md)
- **Add new features**: [Extending the Codebase](./EXTENDING.md)
- **Fix issues**: [Troubleshooting](./TROUBLESHOOTING.md)
