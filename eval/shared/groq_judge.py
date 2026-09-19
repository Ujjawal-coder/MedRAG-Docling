from __future__ import annotations

import os

from openai import OpenAI
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM
from src.core.settings import get_settings


class GroqJudgeModel(DeepEvalBaseLLM):
    def __init__(self):
        settings = get_settings()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required to run DeepEval."
            )

        self.groq_model = settings.groq_model

        self.client = OpenAI(
            api_key=api_key,
            base_url=settings.groq_base_url,
        )

        super().__init__("groq-judge")

    def load_model(self, *args, **kwargs):
        return self.client

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ):
        kwargs = {
            "model": self.groq_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "deepeval_response",
                    "strict": False,
                    "schema": schema.model_json_schema(),
                },
            }

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty evaluation response."
            )

        if schema is not None:
            return schema.model_validate_json(content)

        return content

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.groq_model