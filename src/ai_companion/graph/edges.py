from langgraph.graph import END
from typing_extensions import Literal, Union

from ai_companion.graph.state import AICompanionState
from ai_companion.settings import settings


def should_summarize_conversation(
    state: AICompanionState,
) -> Union[Literal["summarize_conversation_node"], str]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def select_workflow(
    state: AICompanionState,
) -> Literal["conversation_node", "image_node", "audio_node"]:
    """
    Selects the workflow based on the current state of the AI Companion.

    Args:
        state (AICompanionState): The current state of the AI Companion.

    Returns:
        str: selected workflow node.
    """
    workflow = state["workflow"]

    if workflow == "image":
        return "image_node"

    elif workflow == "audio":
        return "audio_node"

    else:
        return "conversation_node"
