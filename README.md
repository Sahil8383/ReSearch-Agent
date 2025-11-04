# ReSearch-Agent

A powerful ReAct (Reasoning + Acting) agent built from scratch that combines reasoning with web search capabilities. This agent uses Claude (Anthropic) for intelligent reasoning and Tavily for real-time web search.

## Features

- 🤖 **ReAct Pattern**: Implements the Reasoning + Acting loop for intelligent decision-making
- 🔍 **Web Search**: Real-time web search using Tavily API
- 💬 **Streaming Responses**: Real-time streaming of agent thoughts and responses
- 🔄 **Self-Correction**: Agent can realize mistakes and adapt its strategy
- 📝 **Step-by-Step Feedback**: Clear visibility into agent's thinking process

## Architecture

The agent follows a strict ReAct loop:

1. **Think** - Analyze what information is needed
2. **Act** - Execute a single action (web search)
3. **Observe** - Receive results from the action
4. **React** - Process observations and decide next steps
5. **Repeat** - Continue until sufficient information is gathered

## Prerequisites

- Python 3.8+
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- Tavily API key ([Get one here](https://tavily.com/))

## Installation

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

```bash
cp .env.example .env
```

4. Edit `.env` and add your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

Run the agent:

```bash
python main.py
```

The agent will prompt you to enable/disable streaming, then you can start asking questions!

### Example Interaction

```
💬 Your Question: tell me about the latest developments in AI

🔄 ITERATION 1
💭 Agent is thinking...
📋 Agent's Response:
Thought: I need to search for recent AI developments
Action: web_search: latest AI developments 2024
PAUSE

⚡ Executing Action: web_search
🔍 Searching the web...

👁️ Observation Received:
[Search results...]

🔄 ITERATION 2
...
```

## Project Structure

```
first-agent/
├── agent/
│   ├── __init__.py          # Package initialization
│   ├── react_agent.py        # Main ReAct agent implementation
│   ├── tools.py              # Web search tool (Tavily)
│   └── message.py            # Message class for conversation history
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Key Components

### ReActAgent (`agent/react_agent.py`)

- Main agent class implementing the ReAct loop
- Handles LLM interactions (streaming and non-streaming)
- Manages conversation history
- Parses actions and final answers

### WebSearchTool (`agent/tools.py`)

- Wraps Tavily API for web search
- Formats search results for agent consumption

### Message (`agent/message.py`)

- Simple class for storing conversation messages
- Converts messages to API format

## Configuration

You can customize the agent when initializing:

```python
from agent import ReActAgent

agent = ReActAgent(
    model="claude-3-5-haiku-20241022",  # Claude model to use
    max_iterations=10,                  # Maximum loop iterations
    stream=True                         # Enable streaming responses
)
```

## Dependencies

- `anthropic` - Anthropic Python SDK for Claude API
- `python-dotenv` - Environment variable management
- `tavily-python` - Tavily API client for web search

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]

## Acknowledgments

- Built using Anthropic's Claude models
- Web search powered by Tavily
- Implements the ReAct (Reasoning + Acting) pattern
