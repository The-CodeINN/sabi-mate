import os
import tempfile
from typing import Optional

from ai_companion.core.exceptions import SpeechToTextError
from ai_companion.settings import settings
from groq import Groq


class SpeechToText:
    """Handles speech-to-text conversion using the Groq's Whisper API."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self):
        """Initialize the SpeechToText class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[Groq] = None

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

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe the given audio file to text.

        Args:
            audio_data (bytes): The audio data to transcribe.

        Returns:
            str: The transcribed text.

        Raises:
            ValueError: If the audio data is empty or invalid.
            SpeechToTextError: If the transcription fails.
        """
        if not audio_data:
            raise ValueError("Audio data is empty or invalid.")

        try:
            # Create a temporary file to store the audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_file.write(audio_data)
                temp_file_path = temp_audio_file.name

                try:
                    # Open the temporary file for reading
                    with open(temp_file_path, "rb") as audio_file:
                        # Transcribe the audio file using the Groq client
                        transcription = self.client.audio.transcriptions.create(
                            file=audio_file,
                            model=settings.STT_MODEL_NAME,
                            language="en",
                            response_format="text",
                        )

                    if not transcription:
                        raise SpeechToTextError("Transcription failed. No text returned.")

                    return transcription.text
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)

        except SpeechToTextError as e:
            raise e

        except Exception as e:
            raise SpeechToTextError(f"Transcription failed: {str(e)}") from e
