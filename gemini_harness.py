from google import genai
from google.genai import types
import os

class GeminiHarness:
    """
    We sometimes want to be able to use GenAI to generate text.
    This gets stored with the bot class so any cog can use it.
    """
    def __init__(self, api_key: str | None = None, model: str | None = None, thinking_level: str | None = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if model is None:
            model = os.getenv("GEMINI_MODEL")
        if thinking_level is None:
            thinking_level = os.getenv("GEMINI_THINKING_LEVEL")
        
        self.api_key = api_key
        self.model = model
        self.thinking_level = thinking_level

        self.client = genai.Client(
            api_key=self.api_key,
        )

    def generate(self, prompt: str | list[types.Content], response_mime_type: str | None = None, response_schema: types.Schema | None = None):
        if isinstance(prompt, str):
            prompt = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]

        if response_mime_type is None:
            if response_schema is None:
                response_mime_type = "text/plain"
            else:
                response_mime_type = "application/json"

        generate_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level,
            ),
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=generate_config,
        )

        return response