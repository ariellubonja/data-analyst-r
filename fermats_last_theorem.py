import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
import os
import openai


api_key = os.environ.get("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY environment variable.")

# Set API key for OpenAI
openai.api_key = api_key



async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
    )
    team = MagenticOneGroupChat([assistant], model_client=model_client)
    await Console(team.run_stream(task="Provide a different proof for Fermat's Last Theorem"))


asyncio.run(main())