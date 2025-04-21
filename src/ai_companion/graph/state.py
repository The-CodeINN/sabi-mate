from langgraph.graph import MessagesState


class AICompanionState(MessagesState):
    """
    Sate class for the AI Companion workflow.

    Extends the MessagesState to track conversation history and last message recieved.

    Attributes:
        summary (AnyMessage): The last message received in the conversation.
        workflow (str): The current workflow being executed.
        apply_activity (str): The current activity to apply in the workflow.
        image_path (str): The path to the image to be used in the workflow.
        workflow (str): The current workflow being executed.
        audio_buffer (str): The audio buffer to be used for speech-to-text conversion.
        current_activity (str): The current activity of SabiMate based on schedule
        memory_context (str): The context of the memory to be injected into the characted card.
    """

    summary: str
    workflow: str
    audio_buffer: str
    current_activity: str
    memory_context: str
    apply_activity: str
    image_path: str
