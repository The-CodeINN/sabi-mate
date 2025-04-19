import re

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import SecretStr

from ai_companion.modules.image.image_to_text import ImageToText
from ai_companion.modules.image.text_to_image import TextToImage
from ai_companion.modules.speech import TextToSpeech
from ai_companion.settings import settings


def get_chat_model(temperature: float = 0.7) -> ChatGroq:
    """Get a ChatGroq model instance based on the provided model name."""
    api_key = SecretStr(settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    return ChatGroq(
        model=settings.TEXT_MODEL_NAME,
        api_key=api_key,
        temperature=temperature,
    )


def get_text_to_speech_module():
    return TextToSpeech()


def get_image_to_text_module():
    return ImageToText()


def get_text_to_image_module():
    return TextToImage()


def remove_asterisks_content(text: str) -> str:
    """Remove asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    """Parser to remove asterisks from the output."""

    def parse(self, text: str) -> str:
        """Parse the text and remove asterisks."""
        return remove_asterisks_content(super().parse(text))
