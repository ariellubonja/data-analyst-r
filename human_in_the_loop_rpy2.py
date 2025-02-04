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

    IMPORTANT: When the task is complete, please append the literal token
    TASK_COMPLETE on a new line outside of the code block.
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Assistant Agent for R code generation.")
        self._model_client = model_client
        # Keep a chat history so the model has conversation context.
        # The system message now instructs the model to include TASK_COMPLETE
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=(
                    "You are an R coder agent that's directing an rpy2 terminal to achieve a certain goal. "
                    "Your task is: 1) read dsc.csv from the local folder, compute mean & std dev of 'aorta' column; "
                    "2) do any other data steps needed. "
                    "When you have finished your R code execution successfully, "
                    " write TASK_COMPLETE on a new line outside of the code block. This will return control to the user\n\n"
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
        When we receive a message (from the user or from the Executor),
        we append it to the conversation context, call the LLM, then publish the response.
        """
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)

        # Print the LLM's text to the console
        print("\n================= Assistant says =================\n" + result.content)

        # Store the response in the chat history
        self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))

        # Publish the assistant's response for the Executor to see
        await self.publish_message(Message(result.content), DefaultTopicId())

###############################################################################
# 2) The Executor agent: uses rpy2 to run the R code and detect TASK_COMPLETE
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
    It captures console output via custom callbacks. If the output includes the token
    TASK_COMPLETE then it signals completion.
    """
    def __init__(self, task_complete_event: asyncio.Event = None) -> None:
        super().__init__("R Executor Agent")
        self._task_complete_event = task_complete_event

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        code = extract_r_code(message.content)
        if not code:
            print("\n--------- Executor: NO R CODE FOUND ---------\n")
            return

        result = self.run_r_code_rpy2(code)
        print("\n--------- Executor Output ---------\n" + result)

        # If the result includes the TASK_COMPLETE flag, signal completion.
        if "TASK_COMPLETE" in result:
            print("\n--------- Executor: TASK COMPLETE detected ---------\n")
            if self._task_complete_event is not None:
                self._task_complete_event.set()

        # Publish the result so that the Assistant (or other agents) sees it.
        await self.publish_message(Message(result), DefaultTopicId())

    def run_r_code_rpy2(self, code: str) -> str:
        """
        Evaluate the R code in the *same* R session using rpy2.
        Capture console output by overriding rinterface callbacks.
        """
        output_buffer = []

        # Intercept console output by overriding the callbacks in rpy2.rinterface_lib.callbacks.
        from rpy2.rinterface_lib import callbacks

        def console_write_print(x):
            output_buffer.append(x)

        # Save the original callbacks.
        old_writeconsole = callbacks.consolewrite_print
        old_warnerror = callbacks.consolewrite_warnerror

        # Override with our custom function.
        callbacks.consolewrite_print = console_write_print
        callbacks.consolewrite_warnerror = console_write_print

        try:
            ro.r(code)  # Execute the R code.
        except Exception as e:
            msg = f"ERROR:\n{str(e)}"
        else:
            joined = "".join(output_buffer)
            # Check for any R errors in the captured output.
            if "Error" in joined:
                msg = f"ERROR:\n{joined}"
            else:
                msg = f"SUCCESS:\n{joined}"
        finally:
            # Restore the original callbacks.
            callbacks.consolewrite_print = old_writeconsole
            callbacks.consolewrite_warnerror = old_warnerror

        return msg

###############################################################################
# 3) A simple loop for human feedback: user can type instructions or corrections
###############################################################################
async def main():
    # logging.basicConfig(level=logging.DEBUG)

    # Ensure you have a valid OpenAI key or set a fake default.
    openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-FAKEKEY")

    # Create an asyncio event that will be set when the task is complete.
    task_complete_event = asyncio.Event()

    # We'll create a local runtime.
    runtime = SingleThreadedAgentRuntime()

    # Register the Assistant and Executor.
    async def assistant_factory():
        return Assistant(
            model_client=OpenAIChatCompletionClient(model="gpt-4o-mini")
        )

    async def executor_factory():
        # Pass the task_complete_event to the Executor.
        return Executor(task_complete_event=task_complete_event)

    await Assistant.register(runtime, "assistant", assistant_factory)
    await Executor.register(runtime, "executor", executor_factory)

    runtime.start()

    # Start with an initial user request.
    initial_task = (
        "Compute mean and std dev from the 'aorta' column in dsc.csv. Then say 'TASK_COMPLETE' when done."
    )
    print(f"\n---------- user ----------\n{initial_task}")
    await runtime.publish_message(Message(initial_task), DefaultTopicId())

    # Main feedback loop.
    while True:
        # Wait for idle: when no pending messages remain.
        await runtime.stop_when_idle()

        # If the TASK_COMPLETE flag has been set, let the user know and optionally exit.
        if task_complete_event.is_set():
            print("\n=== TASK COMPLETE ===\n")
            # Optionally, you can break out of the loop if no further feedback is needed:
            # break

        # Otherwise, ask the user for additional feedback.
        cont = input("\nTEAM is idle. Enter feedback or corrections (empty to exit): ")
        if not cont.strip():
            print("\nExiting...\n")
            break
        print(f"\n---------- user ----------\n{cont}")
        runtime.start()
        await runtime.publish_message(Message(cont), DefaultTopicId())

    await runtime.stop_when_idle()
    print("\nAll done.\n")

if __name__ == "__main__":
    asyncio.run(main())
