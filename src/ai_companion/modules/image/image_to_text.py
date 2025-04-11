import base64
import logging
import os
from typing import Optional, Union

from ai_companion.core.exceptions import ImageToTextError
from ai_companion.settings import settings
from groq import Groq
from groq.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)


class ImageToText:
    """Handles image-to-text conversion using the Groq's Whisper API."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self):
        """Initialize the ImageToText class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[Groq] = None
        self._logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Check if required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> Groq:
        """Lazy load the Groq client."""
        if not self._client:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client

    async def analyse_image(self, image_data: Union[str, bytes], prompt: str = "") -> str:
        """Analyze the provided image data and return the extracted text.

        Args:
            image_data (Union[str, bytes]): The image data to analyze. Can be a base64 string or raw bytes.
            prompt (str): Optional prompt to guide the image analysis.


        Returns:
            str: The description or analysis of the image.

        """
        try:
            # Handle file path
            if isinstance(image_data, str):
                if not os.path.isfile(image_data):
                    raise ValueError("Provided image path does not exist.")
                with open(image_data, "rb") as image_file:
                    image_bytes = image_file.read()
            else:
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data cannot be empty.")

            # Convert image bytes to base64 string
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Create message for the Groq API

            content = [
                ChatCompletionContentPartTextParam(text=prompt, type="text"),
                ChatCompletionContentPartImageParam(
                    image_url={"url": f"data:image/jpeg;base64,{image_base64}"}, type="image_url"
                ),
            ]

            messages = [ChatCompletionUserMessageParam(role="user", content=content)]

            # Make the API call
            response = self.client.chat.completions.create(
                model=settings.ITT_MODEL_NAME,
                messages=messages,
                max_tokens=1000,
            )

            if not response:
                raise ImageToTextError("No response from the image analysis API.")

            description = response.choices[0].message.content or ""
            self._logger.info(f"Image analysis result: {description}")

            return description

        except Exception as e:
            self._logger.error(f"Error during image analysis: {str(e)}")
            raise ImageToTextError(f"Failed to analyze image: {str(e)}") from e
