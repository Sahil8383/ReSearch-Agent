# ReAct Agent Architecture

This document explains the architecture, design, and operation of the ReAct (Reasoning + Acting) agent implementation.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [ReAct Process](#react-process)
- [Capabilities](#capabilities)
- [Memory System](#memory-system)
- [Tools](#tools)
- [Usage Examples](#usage-examples)

## Overview

The ReAct Agent is an intelligent AI agent that combines **reasoning** with **acting** to solve complex tasks. It follows a structured loop where it thinks about what to do, takes actions, observes the results, and reacts accordingly until it can provide a final answer.

### Key Features

- 🤖 **ReAct Pattern**: Implements the Reasoning + Acting loop for intelligent decision-making
- 🔍 **Web Search**: Real-time web search using Tavily API
- 🐍 **Code Execution**: Safe Python code execution for calculations and data processing
- 💾 **Short-term Memory**: Maintains context from the last 5 messages
- 🔄 **Iterative Problem Solving**: Can perform multiple actions in sequence to solve complex problems
- 📊 **Action Tracking**: Tracks all actions taken during problem-solving

## Architecture

The agent is built with a modular architecture consisting of three main components:

```
agent/
├── react_agent.py    # Core ReAct agent implementation
├── message.py        # Message handling and memory management
└── tools.py          # External tools (web search, code execution)
```

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ReActAgent                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  System Prompt (Defines behavior & rules)        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Message    │  │   Tools      │  │   LLM        │ │
│  │   Manager    │  │   Manager    │  │   Client     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                 │                  │          │
│         └─────────────────┴──────────────────┘          │
│                        │                                │
│              ┌─────────▼─────────┐                      │
│              │  ReAct Loop       │                      │
│              │  (Think→Act→Observe)                     │
│              └────────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ReActAgent (`react_agent.py`)

The main agent class that orchestrates the ReAct loop.

**Key Responsibilities:**

- Manages the conversation history
- Executes the ReAct loop (think → act → observe → react)
- Parses LLM responses to extract actions and final answers
- Coordinates tool execution
- Handles streaming and non-streaming modes

**Key Methods:**

- `run(query)`: Main entry point that executes the ReAct loop
- `call_llm()`: Calls the Anthropic Claude API
- `parse_action(text)`: Extracts actions from LLM responses
- `execute_action(action_type, action_input)`: Executes tools
- `is_final_answer(text)`: Detects when agent provides final answer

**Configuration:**

- `model`: Claude model to use (default: `claude-3-5-haiku-20241022`)
- `max_iterations`: Maximum loop iterations (default: 10)
- `stream`: Enable streaming responses (default: True)
- `temperature`: LLM temperature (0.3 for focused responses)

### 2. Message (`message.py`)

Manages conversation history and short-term memory.

**Key Features:**

- **Short-term Memory**: Automatically maintains the last 5 messages
- **Class-level Storage**: Memory is shared across agent instances
- **Database Integration**: Can save/load memory from database
- **Automatic Management**: Messages are automatically added to memory

**Key Methods:**

- `__init__(role, content)`: Creates a message and adds to memory
- `get_short_term_memory()`: Retrieves last 5 messages
- `save_to_db()`: Converts memory to database format
- `load_from_db(db_memory)`: Restores memory from database
- `clear_memory()`: Clears short-term memory
- `set_memory_size(size)`: Adjusts memory size

**Memory Behavior:**

- New messages are automatically added to memory
- Memory is limited to the last N messages (default: 5)
- Memory persists across agent runs (when using database)
- Memory can be cleared or reset as needed

### 3. Tools (`tools.py`)

Provides external capabilities for the agent.

#### WebSearchTool

Performs web searches using the Tavily API.

**Capabilities:**

- Real-time web search
- Returns formatted results with titles, snippets, and URLs
- Configurable number of results (default: 5)

**Usage:**

```python
tool = WebSearchTool()
results = tool.search("latest AI news")
```

**Requirements:**

- `TAVILY_API_KEY` environment variable
- `tavily-python` package installed

#### CodeExecutionTool

Safely executes Python code in an isolated environment.

**Capabilities:**

- Execute arbitrary Python code
- Capture stdout and stderr
- Handle syntax and runtime errors
- Timeout protection (default: 10 seconds)
- Isolated execution namespace

**Safety Features:**

- Isolated namespace (prevents affecting main process)
- Error handling for syntax and runtime errors
- Output capture (stdout/stderr)
- Timeout protection

**Usage:**

```python
tool = CodeExecutionTool()
result = tool.execute("print('Hello, World!')")
```

**Limitations:**

- Code runs in isolated namespace
- No access to main process variables
- Timeout enforced (default: 10 seconds)

## ReAct Process

The agent follows a strict ReAct (Reasoning + Acting) loop:

### Process Flow

```
1. User Query
   ↓
2. Agent Thinks (LLM generates thought)
   ↓
3. Agent Acts (parses action from thought)
   ↓
4. Tool Execution (web_search or execute_code)
   ↓
5. Observation (tool result)
   ↓
6. Agent Reacts (incorporates observation)
   ↓
7. Repeat (steps 2-6) OR Final Answer
```

### Detailed Step-by-Step Process

#### Step 1: Initialization

- User provides a query
- Agent resets conversation history
- Query is added as a user message

#### Step 2: Thinking Phase

- Agent calls LLM with system prompt and conversation history
- LLM generates a response following the format:
  ```
  Thought: <reasoning about what to do>
  Action: <action_type>: <action_input>
  PAUSE
  ```
- Or if ready to answer:
  ```
  Thought: <reasoning based on observations>
  Final Answer: <complete answer>
  ```

#### Step 3: Action Parsing

- Agent parses the LLM response
- Extracts action type (`web_search` or `execute_code`)
- Extracts action input (query or code)
- If no action found, checks for final answer

#### Step 4: Action Execution

- Agent executes the appropriate tool
- For `web_search`: Calls Tavily API with query
- For `execute_code`: Executes Python code in isolated environment
- Tool returns observation (results or output)

#### Step 5: Observation Integration

- Observation is formatted and added to conversation
- Format: `"Observation from <action_type>: <observation>"`
- Added as a user message (simulating user feedback)

#### Step 6: Reaction

- Agent continues loop with updated context
- Can perform additional actions if needed
- Or provide final answer if sufficient information gathered

#### Step 7: Termination

- Loop continues until:
  - Agent provides final answer, OR
  - Maximum iterations reached (default: 10)
- Final answer is extracted and returned

### System Prompt Rules

The agent follows strict rules defined in the system prompt:

1. **One Action Per Response**: Agent can only take ONE action per LLM call
2. **No Premature Answers**: Cannot provide final answer without observations
3. **Strict Format**: Must follow `Thought → Action → PAUSE` or `Thought → Final Answer`
4. **Wait for Observations**: Must wait for tool results before continuing
5. **No Self-Generated Observations**: Cannot generate its own observations

### Example Execution Flow

```
User: "What is the weather in San Francisco?"

Iteration 1:
  Thought: "I need to search for current weather in San Francisco"
  Action: web_search: current weather San Francisco
  PAUSE
  → Observation: [Search results with weather data]

Iteration 2:
  Thought: "Based on the search results, I have the weather information"
  Final Answer: "The current weather in San Francisco is..."
```

## Capabilities

### What the Agent Can Do

1. **Answer Questions**

   - Uses web search to find current information
   - Synthesizes information from multiple sources
   - Provides comprehensive answers

2. **Perform Calculations**

   - Executes Python code for mathematical operations
   - Can process data and perform analysis
   - Returns computed results

3. **Research Topics**

   - Searches the web for information
   - Can perform multiple searches to gather comprehensive data
   - Combines information from various sources

4. **Solve Multi-Step Problems**

   - Breaks down complex problems into steps
   - Performs sequential actions
   - Maintains context across iterations

5. **Code Execution**
   - Runs Python code snippets
   - Performs data processing
   - Executes calculations and algorithms

### Limitations

1. **Maximum Iterations**: Limited to 10 iterations by default (configurable)
2. **Tool Availability**: Requires API keys for web search (Tavily)
3. **Code Execution**: Runs in isolated environment, no access to main process
4. **Memory**: Only maintains last 5 messages in short-term memory
5. **Action Types**: Currently supports only `web_search` and `execute_code`

## Memory System

### Short-Term Memory

The agent maintains a **short-term memory** of the last 5 messages (configurable).

**Purpose:**

- Provides context across multiple agent runs
- Maintains conversation continuity
- Can be persisted to database

**How It Works:**

- Messages are automatically added to memory when created
- Memory is stored at the class level (shared across instances)
- Old messages are automatically removed when limit is reached
- Memory can be saved to/loaded from database

**Memory Lifecycle:**

```
Message Created → Added to Memory → (if > 5 messages) → Oldest Removed
```

### Conversation History

In addition to short-term memory, the agent maintains:

- **Current Conversation**: Messages from the current agent run
- **Short-Term Memory**: Last 5 messages from previous runs (if persisted)

When calling the LLM, both are combined:

1. Short-term memory (context from previous runs)
2. Current conversation (current run messages)

This provides both continuity and current context.

## Tools

### Available Tools

#### 1. Web Search (`web_search`)

**Purpose**: Search the web for real-time information

**Input Format**:

```
Action: web_search: <search query>
```

**Output Format**:

```
1. <Title>
   <Snippet>
   URL: <url>

2. <Title>
   ...
```

**Example**:

```
Action: web_search: latest AI developments 2024
```

#### 2. Code Execution (`execute_code`)

**Purpose**: Execute Python code for calculations or data processing

**Input Format**:

```
Action: execute_code: <python code>
PAUSE
```

**Output Format**:

```
Output:
<stdout content>

Errors:
<stderr content>

Result: <last expression result (if applicable)>
```

**Example**:

```
Action: execute_code:
result = 2 + 2
print(f"Result: {result}")
PAUSE
```

**Safety**:

- Code runs in isolated namespace
- No access to main process variables
- Timeout protection (10 seconds)
- Error handling for syntax/runtime errors

## Usage Examples

### Basic Usage

```python
from agent import ReActAgent

# Create agent instance
agent = ReActAgent(
    model="claude-3-5-haiku-20241022",
    max_iterations=10,
    stream=True
)

# Run agent with a query
answer = agent.run("What is the capital of France?")
print(answer)
```

### Non-Streaming Mode

```python
agent = ReActAgent(stream=False)
answer = agent.run("Calculate 15 * 23")
```

### Custom Configuration

```python
agent = ReActAgent(
    model="claude-3-5-sonnet-20241022",  # Use Sonnet model
    max_iterations=15,                    # Allow more iterations
    stream=False                          # Disable streaming
)
```

### Memory Management

```python
from agent import Message, ReActAgent

# Clear memory
Message.clear_memory()

# Set custom memory size
Message.set_memory_size(10)

# Get current memory
memory = Message.get_short_term_memory()

# Save to database format
db_format = Message.save_to_db()

# Load from database
Message.load_from_db(db_format)
```

### Manual Message Management

```python
agent = ReActAgent()

# Add messages manually
agent.add_message("user", "Hello")
agent.add_message("assistant", "Hi there!")

# Get messages for API
messages = agent.get_messages_for_api(use_short_term_memory=True)

# Reset agent
agent.reset(clear_short_term_memory=False)
```

## Integration with API

The agent is designed to work with the FastAPI wrapper (`api/agent_wrapper.py`):

- **Synchronous Execution**: Direct `run()` calls
- **Streaming Support**: Token-by-token streaming via SSE
- **Session Management**: Memory persistence via database
- **Action Tracking**: All actions are logged and returned

See the main `README.md` for API usage examples.

## Design Decisions

### Why ReAct Pattern?

The ReAct (Reasoning + Acting) pattern provides:

- **Transparency**: Clear reasoning process visible to users
- **Controllability**: Can interrupt and guide the process
- **Debugging**: Easy to see where and why actions were taken
- **Reliability**: Structured approach reduces errors

### Why Short-Term Memory?

- **Context Continuity**: Maintains conversation context across runs
- **Efficiency**: Only keeps relevant recent messages
- **Flexibility**: Can be cleared or adjusted as needed
- **Persistence**: Can be saved to database for session management

### Why Strict Action Format?

- **Reliability**: Prevents LLM from taking multiple actions at once
- **Parsing**: Makes action extraction reliable
- **Control**: Ensures agent follows expected behavior
- **Debugging**: Easier to trace agent decisions

## Future Enhancements

Potential improvements:

- Additional tools (file operations, API calls, etc.)
- Long-term memory with embeddings
- Tool result caching
- Parallel action execution
- Custom tool registration
- Enhanced error recovery
- Action planning ahead of execution

## Dependencies

- `anthropic`: Claude API client
- `tavily-python`: Web search API
- `python-dotenv`: Environment variable management

See `requirements.txt` for complete dependency list.
