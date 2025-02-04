import re
import asyncio
import os
import logging
from typing import List
from dataclasses import dataclass

# rpy2 for direct R evaluation
import rpy2.robjects as ro
import rpy2.rinterface as rinterface

# Base classes from autogen_core
from autogen_core import (
    SingleThreadedAgentRuntime,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler
)
# LLM message types
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    LLMMessage
)

# We'll use OpenAI as a model client
from autogen_ext.models.openai import OpenAIChatCompletionClient
import openai

###############################################################################
# A simple "Message" class representing communication between agents
###############################################################################
@dataclass
class Message:
    content: str

###############################################################################
# 1) The Assistant agent: writes R code in triple backticks
###############################################################################
@default_subscription
class Assistant(RoutedAgent):
    """
    This agent uses a chat model to produce R code in a single code-fence:
    ```r
    # code
    ```
    Then it publishes that code to the 'Executor' agent.
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Assistant Agent for R code generation.")
        self._model_client = model_client
        # Keep a chat history so the model has conversation context
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=(
                    "You are an R coder agent. "
                    "Your task is: 1) read dsc.csv from the local folder, compute mean & std dev of 'aorta' column; "
                    "2) do any other data steps needed. Once you succeed, say 'DONE'.\n\n"
                    "IMPORTANT:\n"
                    " - Always produce exactly one code block with:\n"
                    "```r\n# R code\n```\n"
                    " - If there's an error from the Executor, revise your code and try again.\n"
                    " - Summaries or explanations go outside the code block.\n"
                )
            )
        ]

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        """
        When we receive a message (from user or from the Executor),
        we append it as user context, call the LLM, then publish the LLM response.
        """
        # The message is "user" in a conversation sense
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)

        # Print the LLM's text to console
        print("\n================= Assistant says =================\n" + result.content)

        # Store in chat history
        self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))

        # Publish it so the Executor sees it
        await self.publish_message(Message(result.content), DefaultTopicId())

###############################################################################
# 2) The Executor agent: uses rpy2 to run the R code
###############################################################################
def extract_r_code(full_text: str) -> str | None:
    """
    Looks for a single code-fence: ```r ... ```
    Returns the code inside, or None if not found.
    """
    pattern = re.compile(r"```r\s*\n([\s\S]*?)```")
    match = pattern.search(full_text)
    if match:
        return match.group(1)
    return None

@default_subscription
class Executor(RoutedAgent):
    """
    This agent extracts R code from the Assistant's message and runs it via rpy2.
    We capture console output in a custom callback. If there's an R error, we return 'ERROR:\n...'
    Otherwise 'SUCCESS:\n...'.
    """
    def __init__(self) -> None:
        super().__init__("R Executor Agent")

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        code = extract_r_code(message.content)
        if not code:
            print("\n--------- Executor: NO R CODE FOUND ---------\n")
            return

        result = self.run_r_code_rpy2(code)
        print("\n--------- Executor Output ---------\n" + result)

        # Publish the result so the Assistant sees it
        await self.publish_message(Message(result), DefaultTopicId())

    def run_r_code_rpy2(self, code: str) -> str:
        """
        Evaluate the R code in the *same* R session using rpy2.
        We'll capture console output by overriding rinterface callbacks.
        """
        output_buffer = []

        # We'll intercept console output by overriding the callbacks in rpy2.rinterface_lib.callbacks
        from rpy2.rinterface_lib import callbacks

        def console_write_print(x):
            output_buffer.append(x)

        # Save the original callbacks
        old_writeconsole = callbacks.consolewrite_print
        old_warnerror = callbacks.consolewrite_warnerror

        # Override with our custom function
        callbacks.consolewrite_print = console_write_print
        callbacks.consolewrite_warnerror = console_write_print

        try:
            ro.r(code)  # Execute the R code
        except Exception as e:
            msg = f"ERROR:\n{str(e)}"
        else:
            joined = "".join(output_buffer)
            # Check for any R errors in the captured output
            if "Error" in joined:
                msg = f"ERROR:\n{joined}"
            else:
                msg = f"SUCCESS:\n{joined}"
        finally:
            # Restore the original callbacks
            callbacks.consolewrite_print = old_writeconsole
            callbacks.consolewrite_warnerror = old_warnerror

        return msg


###############################################################################
# 3) A simple loop for human feedback: user can type instructions or corrections
###############################################################################
async def main():
    logging.basicConfig(level=logging.INFO)

    # Ensure you have a valid OpenAI key or set a fake default
    openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-FAKEKEY")

    # We'll create a local runtime
    runtime = SingleThreadedAgentRuntime()

    # Register the Assistant and Executor
    # We'll use GPT-3.5 or GPT-4, whichever is accessible
    async def assistant_factory():
        return Assistant(
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini")
        )

    async def executor_factory():
        return Executor()

    await Assistant.register(runtime, "assistant", assistant_factory)
    await Executor.register(runtime, "executor", executor_factory)

    runtime.start()

    # Start with an initial user request:
    initial_task = (
        "Compute mean and std dev from the 'aorta' column in dsc.csv. Then say 'DONE'."
    )
    print(f"\n---------- user ----------\n{initial_task}")
    await runtime.publish_message(Message(initial_task), DefaultTopicId())

    # Now let's allow the user to correct or continue
    while True:
        # We'll wait for idle, meaning no pending messages
        # That means the assistant & executor have responded
        await runtime.stop_when_idle()
        # If the assistant's code is all done or you want to continue, let's ask
        cont = input("\nTEAM is idle. Enter feedback or corrections (empty to exit): ")
        if not cont.strip():
            print("\nExiting...\n")
            break
        # Otherwise, let's feed that back as user input
        print(f"\n---------- user ----------\n{cont}")
        runtime.start()
        await runtime.publish_message(Message(cont), DefaultTopicId())

    await runtime.stop_when_idle()
    print("\nAll done.\n")

if __name__ == "__main__":
    asyncio.run(main())
