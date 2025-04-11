import base64
import logging
import os
from typing import Optional

from ai_companion.core.exceptions import TextToImageError
from ai_companion.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from ai_companion.settings import settings

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr
from together import Together


class ScenarioPrompt(BaseModel):
    """Scenario prompt for image generation."""

    narrative: str = Field(
        ...,
        description="The AI's narrative or description of the image to be generated.",
    )
    image_prompt: str = Field(
        ...,
        description="The visual prompt to generate an image representing the scene",
    )


class EnhancedPrompt(BaseModel):
    """Enhanced prompt for image generation."""

    content: str = Field(
        ...,
        description="The enhanced text prompt to generate an image.",
    )


class TextToImage:
    """Handles text-to-image generation using the Together API.


    This class is responsible for generating images based on text prompts using the Together API.
    It validates required environment variables and provides methods for image generation and scenario creation.

    """

    # Required environment variables
    REQUIRED_ENV_VARS = ["TOGETHER_API_KEY", "GROQ_API_KEY"]

    def __init__(self):
        """Initialize the TextToImage class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[Together] = None
        self._logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Check if required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        self._logger.info("All required environment variables are set.")

    @property
    def together_client(self) -> Together:
        """Lazy load the Together client."""
        if self._client is None:
            self._client = Together(api_key=settings.TOGETHER_API_KEY)
        return self._client

    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image from the given prompt and save it to the specified output path."""

        if not prompt:
            raise ValueError("Prompt is empty or invalid.")

        try:
            # Generate the image using the Together API
            response = self.together_client.images.generate(
                prompt=prompt,
                model=settings.TTI_MODEL_NAME,
                width=1024,
                height=768,
                steps=4,
                n=1,
                response_format="b64_json",
            )

            # Check if response contains valid data
            if not response or not hasattr(response, "data") or not response.data:
                raise TextToImageError("Invalid or empty response from image generation API")

            # Save the image to the specified output path
            if not hasattr(response.data[0], "b64_json") or not response.data[0].b64_json:
                raise TextToImageError("Response does not contain expected b64_json data")

            image_data = base64.b64decode(str(response.data[0].b64_json))

            if not image_data:
                raise TextToImageError("Generated image data is empty.")

            if output_path:
                with open(output_path, "wb") as image_file:
                    image_file.write(image_data)
                    self._logger.info(f"Image saved to {output_path}")

            self._logger.info("Image generated and saved successfully.")
            return image_data

        except Exception as e:
            self._logger.error(f"Error generating image: {e}")
            raise TextToImageError(f"Error generating image: {e}")

    async def create_scenario(self, chat_history: Optional[list] = None) -> ScenarioPrompt:
        """Create a first-person narrative scenario for image generation based on chat history."""

        try:
            if chat_history is None or len(chat_history) == 0:
                formatted_history = ""
            else:
                formatted_history = "\n".join(
                    [f"{msg.type.title()}: {msg.content}" for msg in chat_history[-5:]]
                )

            self._logger.info(f"Chat history for scenario creation: {formatted_history}")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=SecretStr(settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None,
                temperature=0.4,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(ScenarioPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT,
                )
                | structured_llm
            )

            scenario = chain.invoke({"chat_history": formatted_history})
            self._logger.info(f"Generated scenario: {scenario}")

            # Ensure we return a proper ScenarioPrompt instance
            if isinstance(scenario, dict):
                return ScenarioPrompt(**scenario)
            else:
                return ScenarioPrompt.model_validate(scenario)

        except Exception as e:
            self._logger.error(f"Error creating scenario: {e}")
            raise TextToImageError(f"Error creating scenario: {e}")

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance the given prompt using the Groq API."""

        try:
            if not prompt:
                raise ValueError("Prompt is empty or invalid.")

            self._logger.info(f"Original prompt: {prompt}")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=SecretStr(settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None,
                temperature=0.25,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(EnhancedPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT,
                )
                | structured_llm
            )

            enhanced_prompt = chain.invoke({"prompt": prompt})
            self._logger.info(f"Enhanced prompt: {enhanced_prompt}")

            # Extract the content field from the EnhancedPrompt object
            if isinstance(enhanced_prompt, dict):
                return enhanced_prompt.get("content", "")
            else:
                # Ensure we're working with a valid EnhancedPrompt model instance
                validated_prompt = EnhancedPrompt.model_validate(enhanced_prompt)
                return validated_prompt.content

        except Exception as e:
            self._logger.error(f"Error enhancing prompt: {e}")
            raise TextToImageError(f"Error enhancing prompt: {e}")
