# ReAct Agent API

A powerful FastAPI-based ReAct (Reasoning + Acting) agent API that combines intelligent reasoning with web search capabilities. This API uses Claude (Anthropic) for reasoning and Tavily for real-time web search, with full support for streaming responses and session management.

## Features

- 🤖 **ReAct Pattern**: Implements the Reasoning + Acting loop for intelligent decision-making
- 🔍 **Web Search**: Real-time web search using Tavily API
- 💬 **Streaming Responses**: Real-time streaming of agent thoughts and responses via Server-Sent Events (SSE)
- 🔄 **Session Management**: Persistent conversation sessions with history
- 📝 **Action Tracking**: Detailed tracking of all actions taken by the agent
- 🛡️ **Error Handling**: Comprehensive error handling with structured error responses
- 🔐 **Authentication**: Bearer token authentication (simplified implementation)

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [API Routes](#api-routes)
- [Schemas](#schemas)
- [Error Handling](#error-handling)
- [Streaming](#streaming)
- [Next.js Integration](#nextjs-integration)
- [Database Models](#database-models)

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- Tavily API key ([Get one here](https://tavily.com/))

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd first-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```env
# Database
DATABASE_URL=postgresql+asyncpg://username:password@localhost/dbname

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# Agent Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
AGENT_MODEL=claude-3-5-haiku-20241022
AGENT_MAX_ITERATIONS=10
```

4. Initialize the database:
```bash
python init_db.py
```

5. Run the API:
```bash
python main.py
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://rbmbp33@localhost/first-agent` |
| `API_HOST` | API host address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `CORS_ORIGINS` | Allowed CORS origins (JSON array) | `["http://localhost:3000"]` |
| `ANTHROPIC_API_KEY` | Anthropic API key (required) | - |
| `TAVILY_API_KEY` | Tavily API key (required) | - |
| `AGENT_MODEL` | Claude model to use | `claude-3-5-haiku-20241022` |
| `AGENT_MAX_ITERATIONS` | Maximum agent loop iterations | `10` |

## API Routes

### Base URL
All API routes are prefixed with `/api` unless otherwise specified.

### Health Check

#### `GET /api/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "database": "connected",
  "version": "1.0.0"
}
```

### Chat Endpoints

#### `POST /api/chat`
Send a query to the agent and get a response.

**Headers:**
```
Authorization: Bearer <user_id>
```

**Request Body:**
```json
{
  "query": "What is the latest news about AI?",
  "stream": false,
  "max_iterations": 10
}
```

**Response:**
```json
{
  "answer": "Based on my research...",
  "iterations": 3,
  "status": "completed",
  "actions_taken": [
    {
      "type": "web_search",
      "input": "latest AI news 2024",
      "observation": "Search results...",
      "timestamp": "2024-01-01T00:00:00"
    }
  ],
  "execution_time_ms": 2500,
  "message": "Success"
}
```

**Status Codes:**
- `200`: Success
- `401`: Unauthorized
- `422`: Validation error
- `500`: Internal server error

#### `GET /api/chat/stream`
Stream agent execution with real-time updates using Server-Sent Events (SSE).

**Headers:**
```
Authorization: Bearer <user_id>
```

**Query Parameters:**
- `query` (string, required): The question or task for the agent
- `session_id` (string, optional): Session ID to continue conversation

**Response:** Server-Sent Events stream

**Event Types:**
- `start`: Agent started
- `iteration_start`: New iteration began
- `thinking`: Agent is thinking
- `thought_start`: Thought generation started
- `thought_token`: Token from agent's thought (streaming)
- `thought_complete`: Thought generation completed
- `action`: Action detected
- `observation`: Action observation received
- `final_answer_start`: Final answer generation started
- `final_answer_token`: Token from final answer (streaming)
- `final_answer_complete`: Final answer completed
- `max_iterations`: Maximum iterations reached
- `error`: Error occurred
- `end`: Stream completed

**Example Event:**
```
data: {"type":"start","message":"Starting agent...","query":"What is AI?"}

data: {"type":"iteration_start","iteration":1,"message":"Starting iteration 1"}

data: {"type":"thought_token","token":"I","iteration":1}

data: {"type":"thought_token","token":" need","iteration":1}

data: {"type":"action","action_type":"web_search","input":"AI definition","iteration":1}

data: {"type":"observation","action_type":"web_search","observation":"AI is...","iteration":1}

data: {"type":"final_answer_complete","answer":"AI is...","iterations":2,"status":"completed"}

data: {"type":"end","iterations":2,"status":"completed","actions_count":1}
```

### Session Endpoints

#### `POST /api/sessions`
Create a new session.

**Headers:**
```
Authorization: Bearer <user_id>
```

**Request Body:**
```json
{
  "title": "AI Research Session"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "AI Research Session",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00",
  "conversation_count": 0
}
```

#### `GET /api/sessions/{session_id}`
Get details of a specific session.

**Headers:**
```
Authorization: Bearer <user_id>
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "AI Research Session",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00",
  "conversation_count": 5
}
```

**Status Codes:**
- `200`: Success
- `401`: Unauthorized
- `404`: Session not found

#### `DELETE /api/sessions/{session_id}`
Delete a session and all its conversations.

**Headers:**
```
Authorization: Bearer <user_id>
```

**Status Codes:**
- `204`: Success (no content)
- `401`: Unauthorized
- `404`: Session not found

## Schemas

### Request Schemas

#### `ChatRequest`
```typescript
{
  query: string;           // 1-5000 characters
  stream?: boolean;        // Default: false
  max_iterations?: number; // Default: 10
}
```

#### `CreateSessionRequest`
```typescript
{
  title?: string;          // Optional session title
}
```

### Response Schemas

#### `ChatResponse`
```typescript
{
  answer: string;
  iterations: number;
  status: string;          // "completed" | "failed" | "timeout"
  actions_taken: ActionTaken[];
  execution_time_ms: number;
  message?: string;
}
```

#### `ActionTaken`
```typescript
{
  type: string;            // "web_search" | "execute_code" | etc.
  input: string;
  observation?: string;
  timestamp: string;       // ISO 8601 datetime
}
```

#### `SessionResponse`
```typescript
{
  id: string;              // UUID
  title?: string;
  created_at: string;      // ISO 8601 datetime
  updated_at: string;      // ISO 8601 datetime
  conversation_count: number;
}
```

#### `ErrorResponse`
```typescript
{
  error: string;           // Error type
  message: string;         // Error message
  status_code: number;
  timestamp: string;       // ISO 8601 datetime
}
```

#### `HealthResponse`
```typescript
{
  status: string;          // "healthy"
  timestamp: string;       // ISO 8601 datetime
  database: string;        // "connected"
  version: string;
}
```

## Error Handling

The API uses a global error handling middleware that catches and formats all errors consistently.

### Error Response Format

All errors follow this structure:
```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "status_code": 500,
  "timestamp": "2024-01-01T00:00:00"
}
```

### Error Types

1. **Validation Error (422)**
   - Triggered by: Invalid request body or parameters
   - Example: Missing required field, invalid data type

2. **HTTP Error (4xx, 5xx)**
   - Triggered by: HTTPException in routes
   - Examples:
     - `401 Unauthorized`: Invalid or missing authentication
     - `404 Not Found`: Resource not found
     - `500 Internal Server Error`: Server-side error

3. **Internal Server Error (500)**
   - Triggered by: Unhandled exceptions
   - Logged with full stack trace

### Error Handling in Frontend

```typescript
try {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userId}`
    },
    body: JSON.stringify({ query: 'Hello' })
  });

  if (!response.ok) {
    const error = await response.json();
    console.error('Error:', error.error, error.message);
    // Handle error based on error.status_code
  }

  const data = await response.json();
} catch (error) {
  console.error('Network error:', error);
}
```

## Streaming

The API supports Server-Sent Events (SSE) for real-time streaming of agent execution.

### Streaming Endpoint

**Endpoint:** `GET /api/chat/stream?query=<your_query>&session_id=<optional>`

**Content-Type:** `text/event-stream`

### Event Format

Each event follows the SSE format:
```
data: <JSON_STRING>

```

### Event Types

| Event Type | Description | Payload |
|------------|-------------|---------|
| `start` | Agent execution started | `{type, message, query}` |
| `iteration_start` | New iteration began | `{type, iteration, message}` |
| `thinking` | Agent is thinking | `{type, message}` |
| `thought_start` | Thought generation started | `{type, iteration}` |
| `thought_token` | Token from thought (streaming) | `{type, token, iteration}` |
| `thought_complete` | Thought completed | `{type, iteration}` |
| `action` | Action detected | `{type, action_type, input, iteration}` |
| `observation` | Action observation | `{type, action_type, observation, iteration}` |
| `final_answer_start` | Final answer started | `{type}` |
| `final_answer_token` | Token from final answer | `{type, token}` |
| `final_answer_complete` | Final answer completed | `{type, answer, iterations, status}` |
| `max_iterations` | Max iterations reached | `{type, message}` |
| `error` | Error occurred | `{type, message}` |
| `end` | Stream completed | `{type, iterations, status, actions_count}` |

### Next.js Streaming Example

```typescript
// app/api/chat/stream/route.ts (Next.js API route)
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('query');
  const sessionId = searchParams.get('session_id');
  const userId = request.headers.get('authorization')?.replace('Bearer ', '');

  const apiUrl = `http://localhost:8000/api/chat/stream?query=${encodeURIComponent(query)}${sessionId ? `&session_id=${sessionId}` : ''}`;
  
  const response = await fetch(apiUrl, {
    headers: {
      'Authorization': `Bearer ${userId}`
    }
  });

  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    }
  });
}
```

```typescript
// Client-side component
'use client';

import { useState, useEffect } from 'react';

export default function ChatStream() {
  const [messages, setMessages] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const handleStream = async (query: string) => {
    setIsStreaming(true);
    setMessages([]);

    const eventSource = new EventSource(
      `/api/chat/stream?query=${encodeURIComponent(query)}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'thought_token':
        case 'final_answer_token':
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last?.type === data.type) {
              return [...prev.slice(0, -1), { ...last, content: last.content + data.token }];
            }
            return [...prev, { type: data.type, content: data.token }];
          });
          break;
        
        case 'action':
          setMessages(prev => [...prev, { 
            type: 'action', 
            content: `Action: ${data.action_type} - ${data.input}` 
          }]);
          break;
        
        case 'observation':
          setMessages(prev => [...prev, { 
            type: 'observation', 
            content: `Observation: ${data.observation}` 
          }]);
          break;
        
        case 'end':
          eventSource.close();
          setIsStreaming(false);
          break;
        
        case 'error':
          console.error('Stream error:', data.message);
          eventSource.close();
          setIsStreaming(false);
          break;
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setIsStreaming(false);
    };
  };

  return (
    <div>
      <button onClick={() => handleStream('What is AI?')}>
        Start Stream
      </button>
      <div>
        {messages.map((msg, i) => (
          <div key={i}>{msg.content}</div>
        ))}
      </div>
    </div>
  );
}
```

## Next.js Integration

### Setup

1. Create a Next.js app:
```bash
npx create-next-app@latest agent-frontend
cd agent-frontend
```

2. Install dependencies:
```bash
npm install
```

### API Client Example

Create `lib/api.ts`:

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ChatRequest {
  query: string;
  stream?: boolean;
  max_iterations?: number;
}

export interface ChatResponse {
  answer: string;
  iterations: number;
  status: string;
  actions_taken: ActionTaken[];
  execution_time_ms: number;
}

export interface ActionTaken {
  type: string;
  input: string;
  observation?: string;
  timestamp: string;
}

export interface SessionResponse {
  id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  conversation_count: number;
}

class ApiClient {
  private getHeaders(userId: string) {
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userId}`
    };
  }

  async chat(request: ChatRequest, userId: string): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: this.getHeaders(userId),
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Chat request failed');
    }

    return response.json();
  }

  async createSession(title: string, userId: string): Promise<SessionResponse> {
    const response = await fetch(`${API_BASE_URL}/api/sessions`, {
      method: 'POST',
      headers: this.getHeaders(userId),
      body: JSON.stringify({ title })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to create session');
    }

    return response.json();
  }

  async getSession(sessionId: string, userId: string): Promise<SessionResponse> {
    const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
      headers: this.getHeaders(userId)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to get session');
    }

    return response.json();
  }

  async deleteSession(sessionId: string, userId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: this.getHeaders(userId)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to delete session');
    }
  }

  createStreamUrl(query: string, sessionId?: string): string {
    const params = new URLSearchParams({ query });
    if (sessionId) params.append('session_id', sessionId);
    return `${API_BASE_URL}/api/chat/stream?${params.toString()}`;
  }
}

export const apiClient = new ApiClient();
```

### Chat Component Example

Create `components/Chat.tsx`:

```typescript
'use client';

import { useState } from 'react';
import { apiClient, ChatResponse } from '@/lib/api';

export default function Chat() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const userId = 'your-user-id'; // Get from auth context

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.chat(
        { query, stream: false, max_iterations: 10 },
        userId
      );
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {response && (
        <div className="response">
          <h3>Answer:</h3>
          <p>{response.answer}</p>
          <div className="metadata">
            <p>Iterations: {response.iterations}</p>
            <p>Status: {response.status}</p>
            <p>Execution Time: {response.execution_time_ms}ms</p>
          </div>
          <div className="actions">
            <h4>Actions Taken:</h4>
            {response.actions_taken.map((action, i) => (
              <div key={i} className="action">
                <strong>{action.type}:</strong> {action.input}
                {action.observation && (
                  <div className="observation">{action.observation}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

### Environment Variables

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Database Models

### User
- `id` (UUID): Primary key
- `email` (String): Unique email address
- `username` (String): Unique username
- `hashed_password` (String): Hashed password
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp
- `is_active` (Boolean): Active status

### Session
- `id` (UUID): Primary key
- `user_id` (UUID): Foreign key to User
- `title` (String, nullable): Session title
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp
- `is_active` (Boolean): Active status
- `short_term_memory` (JSON): Last 5 messages
- `metadata_info` (JSON): Custom metadata

### Conversation
- `id` (UUID): Primary key
- `session_id` (UUID): Foreign key to Session
- `query` (Text): User query
- `response` (Text, nullable): Agent response
- `iterations` (Integer): Number of iterations
- `status` (String): "pending" | "completed" | "failed" | "timeout"
- `error_message` (Text, nullable): Error message if failed
- `actions_taken` (JSON): Array of actions taken
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp
- `completed_at` (DateTime, nullable): Completion timestamp
- `execution_time_ms` (Integer, nullable): Execution time in milliseconds

### AgentLog
- `id` (UUID): Primary key
- `session_id` (UUID, nullable): Foreign key to Session
- `log_level` (String): "DEBUG" | "INFO" | "WARNING" | "ERROR"
- `message` (Text): Log message
- `context` (JSON): Additional context
- `created_at` (DateTime): Creation timestamp

## Project Structure

```
first-agent/
├── agent/                    # Agent core logic
│   ├── __init__.py
│   ├── react_agent.py        # ReAct agent implementation
│   ├── tools.py              # Web search tool
│   └── message.py            # Message class
├── api/                      # FastAPI application
│   ├── __init__.py
│   ├── main.py              # FastAPI app setup
│   ├── config.py            # Configuration
│   ├── database.py          # Database setup
│   ├── dependencies.py     # Dependency injection
│   ├── models.py            # SQLAlchemy models
│   ├── schemas.py           # Pydantic schemas
│   ├── agent_wrapper.py     # Agent wrapper
│   ├── middleware/          # Middleware
│   │   ├── __init__.py
│   │   └── error_handler.py # Error handling
│   ├── routes/              # API routes
│   │   ├── __init__.py
│   │   ├── chat.py          # Chat endpoints
│   │   ├── sessions.py      # Session endpoints
│   │   └── health.py        # Health check
│   └── services/            # Business logic
│       ├── __init__.py
│       └── agent_service.py # Agent service
├── init_db.py               # Database initialization
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]

## Acknowledgments

- Built using Anthropic's Claude models
- Web search powered by Tavily
- Implements the ReAct (Reasoning + Acting) pattern
- FastAPI for the API framework
- PostgreSQL for data persistence
