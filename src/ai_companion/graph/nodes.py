import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import get_character_response_chain, get_router_chain
from ai_companion.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager_async
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.settings import settings


async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke(
        {"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]}
    )
    # Check if response is a BaseModel or a Dict and extract response_type accordingly
    if hasattr(response, "__dict__") and "response_type" in response.__dict__:
        response_type = response.__dict__["response_type"]
    elif isinstance(response, dict) and "response_type" in response:
        response_type = response["response_type"]
    else:
        # Provide a default or raise an appropriate error
        raise ValueError("Response doesn't contain required 'response_type' field")

    return {"workflow": response_type}


async def memory_extraction_node(state: AICompanionState):
    """Extract and store important memories from the conversation.

    This node processes the latest message in the conversation and stores
    any important information in the long-term memory vector store.
    """
    # Skip if there are no messages
    if not state["messages"]:
        return {"next": "router_node"}

    # Get the latest message
    latest_message = state["messages"][-1]

    # Get memory manager with async initialization
    memory_manager = await get_memory_manager_async()

    # Extract and store memories asynchronously
    await memory_manager.extract_and_store_memories(latest_message)

    # Always go to router_node next, which will set the workflow
    return {"next": "router_node"}


async def memory_injection_node(state: AICompanionState):
    """Retrieve relevant memories for the current conversation context."""
    # Skip if there are no messages
    if not state["messages"]:
        return {}

    # Get memory manager with async initialization
    memory_manager = await get_memory_manager_async()

    # Convert all messages to a single string for context lookup
    context = " ".join([str(m.content) for m in state["messages"][-5:] if hasattr(m, "content")])
    memories = await memory_manager.get_relevant_memories_async(context)
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}


def context_injection_node(state: AICompanionState):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False
    return {"apply_activity": apply_activity, "current_activity": schedule_context}


async def summarize_conversation_node(state: AICompanionState):
    """Create a summary of the conversation to reduce token usage."""
    # Using the settings that actually exist
    if len(state["messages"]) < settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return {}

    # Using the chat model to create a summary
    chat_model = get_chat_model()

    # Get all messages to summarize
    messages_to_summarize = state["messages"]
    message_content = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages_to_summarize if hasattr(msg, "content")]
    )

    # Generate the summary
    prompt = f"Summarize the following conversation briefly while preserving key facts and context:\n\n{message_content}"
    summary = await chat_model.ainvoke(prompt)

    # Keep only the most recent messages after summary
    new_messages = state["messages"][-settings.TOTAL_MESSAGES_AFTER_SUMMARY :]

    return {"messages": new_messages, "summary": summary}


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response)}


async def image_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(
        content=f"<image attached by SabiMate generated from prompt: {scenario.image_prompt}>"
    )
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}


async def audio_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    output_audio = await text_to_speech_module.synthesize(response)

    return {"messages": response, "audio_buffer": output_audio}
