# Sample AI Agent using LangGraph and CrewAI

This work is originallybased on [this repo](https://github.com/JoshuaC215/agent-service-toolkit) which was created by [JoshuaC215](https://github.com/JoshuaC215).  Thanks! 

This also uses the great product [Browser Use](https://browseruse.com/) to browse the web.

## Follow Along

Follow along as I post about program on my [websites](https://www.apsquared.co/posts/full-stack-ai-agents), [X account](https://x.com/APSquaredDev) and my Bluesky account [@apsquared](https://bsky.app/profile/apsquared.bsky.social).

Disclaimer: I'm not an expert at Python or LangGraph, just a developer trying to play with AI Agents and hoping to help others.

## Motivation

I wanted to create a simple AI agent that uses LangGraph and Browser Use and can be easily deployed as a full stack application.
I've also been playing with [CrewAI](https://crewai.com/) and wanted to see how to integrate it with LangGraph.

## Overview Diagram

![Overview Diagram](https://apsquared.co/agent_arch.png)

Built with [DiagramGPT](https://www.eraser.io/diagramgpt)

## Client

Part of what I wanted to do is to demonstrate a totally disconnected client that can be used to interact with the agent.  
On our website we have a set of agents that demonstrate a [full stack solution with a NextJS application](https://www.apsquared.co/tools).  You can also see the [code for the client here](https://github.com/apsquared/ap2-agents).

## Try Out the Sample Agents 

### Marketing Agent

The very simple marketing agent can be found [to try on Apquared.co](https://www.apsquared.co/tools/saas-marketing-agent) and helps businesses build a marketing plan based on their website.

### College Finder Agent

The college finder agent can be found [to try on Apquared.co](https://www.apsquared.co/tools/college-finder-agent)

### College Baseball Roster Agent

The [college baseball roster agent](https://www.apsquared.co/tools/team-roster-agent) finds the rosters of a college baseball and analyzes the stats about the players to help prospective players see if they are a fit.

### Vacation House Agent

The [vacation house agent](https://www.apsquared.co/tools/vacation-house-agent) helps find vacation houses based on the user's query.
This agent uses the [crewai](https://crewai.com/) framework to create the agent.

## Settings

Updates settings .in .env and settings.py file (in core)

## Commands

`uv run src/run_service.py` - run as a service

`uv run src/run_agent.py` - run as a single agent

`uv run src/run_agent_stream.py` - run as a single agent with streaming

## Adding new Agents

To add a new agent to the system, follow these steps:

1. Create a new folder in src/agents/
   - Define the agent using LangGraph/LangChain patterns in <agent_name>_agent.py
   - Add type hints and docstrings
   - Create a schema file in the directory <agent_name>_schema.py for agent specific schemas
   - Return the agent object

2. Register the agent in the AGENTS dictionary in `src/agents/agents.py`:
   ```python
   AGENTS = {
       "existing_agent": existing_agent_function,
       "your_new_agent": your_new_agent_function
   }
   ```

3. (Optional) Add any new tools needed by your agent in `src/tools/`:
   - Create new tool functions
   - Add tool configurations
   - Import and include tools in your agent definition

4. (Optional) Add new environment variables in `.env` if required by your agent

5. Update the run_agent.py to specify the new agent and its input:
   ```bash
   uv run src/run_agent.py
   ```

## Additional Tools to Try / Reference Pages

* https://python.langchain.com/docs/integrations/tools/
* https://github.com/dendrite-systems/dendrite-python-sdk
* https://simplescraper.io/docs/api-guide
* https://langchain-ai.github.io/langgraph/concepts/low_level/
* https://app.composio.dev/sdk_guide

## Adding new packages

`uv pip add <package-name>`
`uv pip compile pyproject.toml -o uv.lock --refresh`
`uv pip sync uv.lock`


## DEPLOAY agent to ECS

`./deploy.sh`

DNS name will be displayed verify it is working.  Details for setting up ECS are not provided here.


## API Documentation

### Service Information
- `GET /info` - Get metadata about available agents, models and defaults

### Agent Interaction
- `POST /{agent_id}/invoke` - Invoke an agent synchronously and get final response
- `POST /invoke` - Invoke default agent synchronously
- `POST /{agent_id}/stream` - Stream agent responses including intermediate steps
- `POST /stream` - Stream default agent responses

### Background Agent Management  
- `POST /{agent_id}/start` - Start an agent running in background
- `GET /agent/{run_id}/status` - Get status of background running agent

### Chat History
- `POST /history` - Get chat history for a thread

### Feedback
- `POST /feedback` - Record feedback for an agent run to LangSmith

### Logs
- `GET /logs` - List available log files
- `GET /logs/{filename}` - Get content of specific log file

### Health Check
- `GET /health` - Simple health check endpoint

### Authentication
- All endpoints except `/health` require Bearer token authentication if AUTH_SECRET is set
- Pass token in Authorization header: `Bearer <AUTH_SECRET>`

### Common Parameters
- `thread_id` - Used to maintain conversation context across requests
- `model` - Override default model
- `agent_id` - Specify agent to use (defaults to DEFAULT_AGENT)
- `state` - Initial state for state-based agents
- `message` - Input message for chat-based agents

