import asyncio
import os
from typing import Optional

from ai_companion.core.exceptions import TextToSpeechError
from ai_companion.settings import settings
from elevenlabs import ElevenLabs, Voice, VoiceSettings


class TextToSpeech:
    """Handles text-to-speech conversion using the ElevenLabs API."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

    def __init__(self):
        """Initialize the TextToSpeech class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[ElevenLabs] = None

    def _validate_env_vars(self) -> None:
        """Check if required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        if settings.ELEVENLABS_VOICE_ID is None:
            raise ValueError("ELEVENLABS_VOICE_ID cannot be None")

    @property
    def client(self) -> ElevenLabs:
        """Lazy load the ElevenLabs client."""
        if self._client is None:
            self._client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        return self._client

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous helper method to generate audio using the ElevenLabs API.
        This will be called in a separate thread via asyncio.to_thread."""
        voice_id = settings.ELEVENLABS_VOICE_ID
        if voice_id is None:
            raise TextToSpeechError("ELEVENLABS_VOICE_ID is not set")

        audio_generator = self.client.generate(
            text=text,
            voice=Voice(
                voice_id=voice_id,
                settings=VoiceSettings(stability=0.75, similarity_boost=0.75),
            ),
            model=settings.TTS_MODEL_NAME,
        )

        # Convert the audio generator to bytes
        audio_data = b"".join(audio_generator)
        if not audio_data:
            raise TextToSpeechError("Failed to synthesize speech.")

        return audio_data

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using the ElevenLabs API.

        Args:
            text (str): The text to convert to speech.

        Returns:
            bytes: The audio data of the synthesized speech.

        Raises:
            ValueError: If the text is empty or invalid.
            TextToSpeechError: If the synthesis fails.
        """
        if not text:
            raise ValueError("Text is empty or invalid.")

        try:
            # Use asyncio.to_thread to run the blocking ElevenLabs API call in a separate thread
            audio_data = await asyncio.to_thread(self._synthesize_sync, text)
            return audio_data

        except Exception as e:
            raise TextToSpeechError(f"Failed to synthesize speech: {str(e)}") from e
