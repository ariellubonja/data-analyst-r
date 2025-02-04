import asyncio
import subprocess
import tempfile
from typing import Sequence

from autogen_agentchat.base import Response
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, ChatMessage
from autogen_core import CancellationToken


class RExecutorAgent(BaseChatAgent):
    """
    A custom agent that runs R code snippets it sees in incoming messages.
    Very naive example:
    - If a message has ```r ... ``` code fence, we run it in R.
    - Otherwise, we just echo the text back.
    """

    async def on_messages(
        self,
        messages: Sequence[ChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        if not messages:
            return Response(chat_message=TextMessage(content="No message provided.", source=self.name))

        # We only look at the latest user message:
        last_message = messages[-1]
        if not isinstance(last_message, TextMessage):
            return Response(chat_message=TextMessage(content="Unsupported message type.", source=self.name))

        text = last_message.content
        # Look for fenced R code:
        start_tag = "```r"
        end_tag = "```"

        if start_tag in text and end_tag in text:
            # Extract code:
            start_idx = text.index(start_tag) + len(start_tag)
            end_idx = text.index(end_tag, start_idx)
            r_code = text[start_idx:end_idx].strip()
            # Actually run the code:
            result = self.run_r_code(r_code)
            return Response(
                chat_message=TextMessage(content=f"(R output)\n{result}", source=self.name)
            )
        else:
            # No R code found, just echo the user’s text:
            return Response(
                chat_message=TextMessage(content=f"No R code found. Received:\n{text}", source=self.name)
            )

    def run_r_code(self, r_code: str) -> str:
        """
        Actually runs the R code with a subprocess call to `Rscript`.
        Replace with challengeR/rpy2 if you prefer.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as tmp:
            tmp.write(r_code)
            tmp.flush()
            tmp_name = tmp.name

        try:
            process = subprocess.Popen(
                ["Rscript", tmp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                return f"Error:\n{stderr}"
            return stdout
        except Exception as e:
            return f"Exception running R code: {str(e)}"

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """
        If you need to reset the agent’s internal state,
        do so here. We have no state, so no-op.
        """
        pass

    @property
    def produced_message_types(self):
        # We produce TextMessage, so declare that here.
        return (TextMessage,)

# Quick test:
if __name__ == "__main__":
    async def quick_test():
        agent = RExecutorAgent("RRunner", description="Runs R code")
        # Make up a message that has an R code fence:
        test_msg = TextMessage(content="Hello, here is some code:\n```r\ncat('Hello from R!\n')\n```", source="user")
        resp = await agent.on_messages([test_msg], CancellationToken())
        print("[Agent Output]", resp.chat_message.content)

    asyncio.run(quick_test())
