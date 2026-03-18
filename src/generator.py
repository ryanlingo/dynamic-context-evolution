"""Structured output generation with retries. Supports OpenAI and Anthropic models."""

from __future__ import annotations

import json
import time

from tenacity import retry, stop_after_attempt, wait_exponential

from src.schemas import BatchOutput

# Models that route through the Anthropic API
ANTHROPIC_MODELS = {"claude-haiku-4-5-20251001"}
ANTHROPIC_MODEL_ALIASES = {"claude-haiku-4-5": "claude-haiku-4-5-20251001"}


def _is_anthropic(model: str) -> bool:
    return model in ANTHROPIC_MODELS or model in ANTHROPIC_MODEL_ALIASES


def _resolve_model(model: str) -> str:
    return ANTHROPIC_MODEL_ALIASES.get(model, model)


class IdeaGenerator:
    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self.model = _resolve_model(model)
        self.temperature = temperature
        self.top_p = top_p
        self._is_anthropic = _is_anthropic(model)

        if self._is_anthropic:
            from anthropic import Anthropic
            self.client = Anthropic()
        else:
            from openai import OpenAI
            self.client = OpenAI()

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def generate(self, prompt: str, expected_count: int = 5) -> BatchOutput:
        """Generate a batch of ideas using structured output.

        Retries if the model returns fewer ideas than expected_count.
        Handles rate limits with a 60s sleep before retrying.
        """
        try:
            if self._is_anthropic:
                result = self._generate_anthropic(prompt)
            else:
                result = self._generate_openai(prompt)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
            raise

        if len(result.ideas) < expected_count:
            raise ValueError(
                f"Expected {expected_count} ideas, got {len(result.ideas)}"
            )
        return result

    def _generate_openai(self, prompt: str) -> BatchOutput:
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=BatchOutput,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        completion = self.client.beta.chat.completions.parse(**kwargs)

        if completion.usage:
            self.total_prompt_tokens += completion.usage.prompt_tokens
            self.total_completion_tokens += completion.usage.completion_tokens
        self.total_calls += 1

        return completion.choices[0].message.parsed

    def _generate_anthropic(self, prompt: str) -> BatchOutput:
        schema = BatchOutput.model_json_schema()

        # Build the prompt with JSON schema instruction
        system_prompt = (
            "You are a structured data generator. You MUST respond with valid JSON "
            "matching this schema exactly:\n\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Respond ONLY with the JSON object, no other text."
        )

        kwargs: dict = dict(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        response = self.client.messages.create(**kwargs)

        # Track usage
        if response.usage:
            self.total_prompt_tokens += response.usage.input_tokens
            self.total_completion_tokens += response.usage.output_tokens
        self.total_calls += 1

        # Parse the JSON response
        text = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return BatchOutput.model_validate_json(text)

    @property
    def token_usage(self) -> dict:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_calls": self.total_calls,
        }
