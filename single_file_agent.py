import asyncio
import tempfile
import subprocess
from typing import Sequence

import os
import openai  # only needed if you're using OpenAI keys
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_agentchat.base import Response
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
import logging

# logging.basicConfig(level=logging.DEBUG)


###############################################################################
# 1) Define an RExecutorAgent that runs R code with subprocess.Popen(["Rscript", ...])
###############################################################################
class RExecutorAgent(BaseChatAgent):
    """
    A custom agent that looks for R code in the last message:
      - If we find a code fence like ```r ... ```
      - we extract the code, run it with Rscript, and return the result (stdout or error).
    """

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        if not messages:
            return Response(chat_message=TextMessage(content="(No messages to process.)", source=self.name))

        # We only look at the latest message from the other agent
        latest_msg = messages[-1]
        if not isinstance(latest_msg, TextMessage):
            return Response(chat_message=TextMessage(content="(Unsupported message type.)", source=self.name))

        text = latest_msg.content
        # Very naive extraction of R code from a triple-backtick block ```r ... ```
        code_fence_start = "```r"
        code_fence_end = "```"
        if code_fence_start in text and code_fence_end in text:
            start_idx = text.index(code_fence_start) + len(code_fence_start)
            end_idx = text.index(code_fence_end, start_idx)
            r_code = text[start_idx:end_idx].strip()
            # Actually run that R code:
            result = self.run_r_code(r_code)
            return Response(chat_message=TextMessage(content=result, source=self.name))
        else:
            return Response(chat_message=TextMessage(content="(No R code found in your message. Start your R code block with ```r and end it with ```)", source=self.name))

    def run_r_code(self, code: str) -> str:
        """Run the given R code in a temp file via Rscript. Return 'ERROR:...' or 'SUCCESS:...'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as tmp:
            tmp.write(code)
            tmp.flush()
            tmp_file_name = tmp.name

        try:
            process = subprocess.Popen(
                ["Rscript", tmp_file_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            returncode = process.returncode
            if returncode != 0 or stderr.strip():
                # treat any non-zero exit or any stderr as error
                return f"ERROR:\n{stderr}"
            else:
                return f"SUCCESS:\n{stdout}"
        except Exception as e:
            return f"ERROR:\nException when running R code:\n{str(e)}"

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        # no internal state
        pass

    @property
    def produced_message_types(self):
        return (TextMessage,)


###############################################################################
# 2) Define an RCoderAgent that tries to do:
#    Step 1) read dsc.csv, compute mean+std for a column
#    Step 2) load challengeR library, print its version
#    If an error occurs, revise code until it sees SUCCESS. Then do next step.
#    Finally, say "DONE" after finishing both tasks.
###############################################################################
class RCoderAgent(AssistantAgent):
    """
    A coder-like agent that produces R code in triple-backticks. On error from the R executor,
    it refines and tries again, until it eventually says "DONE".
    """

    def __init__(self, name: str, model_client):
        super().__init__(name=name, model_client=model_client)

        # System prompt: instruct the agent how to do iterative code attempts
        self.system_message = (
            "You are an R coder agent who is guiding an R executor bot which will run your code. The bot will run code it detects in your answer and give you back the results. It will never ask you questions. It is your duty to iteratively write code for the executor to achieve your stated goals. You have two goals:\n"
            "1) Load 'dsc.csv' from the local folder. Compute the mean and std dev of 'aorta'.\n"
            "2) Load 'challengeR' library and print its version.\n\n"

            "IMPORTANT:\n"
            "1. The R executor only understands code in your output when you structure it correctly. So, Always produce exactly ONE fenced code block using the syntax:\n"
            "```r\n# your code\n```\n"
            "2. If you have any normal text or explanation, put it outside that single code block.\n"
            "3. The RExecutor will only run code inside ```r ... ```. If there's no code block, it won't run.\n"
            "4. If you get an ERROR, refine the code and try again.\n"
            "5. Once both tasks succeed, produce a plain text summary and say 'DONE'.\n"
        )


###############################################################################
# 3) Assemble the RoundRobinGroupChat with a termination condition
###############################################################################
async def main():
    # Setup the LLM client (OpenAI GPT-4, or your choice)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY to use OpenAIChatCompletionClient or replace with your model code.")
    openai.api_key = api_key

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    coder = RCoderAgent("RCoder", model_client=model_client)
    r_executor = RExecutorAgent("RExecutor", description="gimi bate")

    # We'll stop once the coder says "DONE" or after 15 messages to avoid infinite loops
    termination = TextMentionTermination("DONE") | MaxMessageTermination(15)

    chat = RoundRobinGroupChat(
        participants=[coder, r_executor],
        termination_condition=termination,
        max_turns=10
    )

    task_prompt = (
        "Please begin. Step 1: compute mean+std from 'aorta' in dsc.csv.\n"
        "Step 2: load challengeR library in R and print its version. Then output your final answer."
    )

    print("==== STARTING ITERATIVE R CODE TASK ====")
    await Console(chat.run_stream(task=task_prompt))


if __name__ == "__main__":
    asyncio.run(main())
