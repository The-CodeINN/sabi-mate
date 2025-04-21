from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from ai_companion.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from ai_companion.core.datetime_utils import get_current_datetime
from ai_companion.graph.utils.helpers import AsteriskRemovalParser, get_chat_model


class RouterResponse(BaseModel):
    response_type: str = Field(
        description="The type of response to be given to user. It must be one of 'conversation', 'image' or 'audio'"
    )


def get_router_chain():
    model = get_chat_model(
        temperature=0.3,
    ).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")],
    )

    return prompt | model


def get_character_response_chain(summary: str = "", timezone: str = "Africa/Lagos"):
    model = get_chat_model()

    # Get current date and time based on specified timezone
    current_datetime = get_current_datetime(timezone)

    # Format the prompt with the current date and time
    system_message = CHARACTER_CARD_PROMPT.format(
        memory_context="{memory_context}",
        current_activity="{current_activity}",
        current_datetime=current_datetime,
    )

    if summary:
        system_message += (
            f"\n\nSummary of conversation earlier between SabiMate and the user: {summary}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), MessagesPlaceholder(variable_name="messages")],
    )

    return prompt | model | AsteriskRemovalParser()
