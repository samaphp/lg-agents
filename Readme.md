# Sample AI Agent using LangGraph

This work is based on [this repo](https://github.com/JoshuaC215/agent-service-toolkit) which was created by [JoshuaC215](https://github.com/JoshuaC215).  Thanks!

This also uses the great product [Browser Use](https://browseruse.com/) to browse the web.

## Follow Along

Follow along as I post about program on my [websites](https://apsquared.co) and my Bluesky account [@apsquared](https://bsky.app/profile/apsquared.bsky.social).

Disclaimer: I'm not an expert at Python or LangGraph, just a developer trying to play with AI Agents and hoping to help others.

## Motivation

I wanted to create a simple AI agent that uses LangGraph and Browser Use and can be easily deployed. 

## Client

Part of what I wanted to do is to demonstrate a totally disconnected client that can be used to interact with the agent.  The client project [can be found here](https://github.com/apsquared/lg-agent-client).

## Settings

Updates settings .in .env and settings.py file (in core)

## Commands

`uv run src/run_service.py` - run as a service

`uv run src/run_agent.py` - run as a single agent

`uv run src/run_agent_stream.py` - run as a single agent with streaming


## Additional Tools to Try

* https://python.langchain.com/docs/integrations/tools/
* https://github.com/dendrite-systems/dendrite-python-sdk
* https://simplescraper.io/docs/api-guide


## Notes to self

UV Help:
https://docs.astral.sh/uv/guides/projects/


Activate venv
`source .venv/bin/activate`

Run service
`python src/run_service.py`



## DEPLOAY agent to ECS

`./deploy.sh`

DNS name will be displayed verify it is working.

http://my-agent-alb-1226714337.us-east-1.elb.amazonaws.com/health

