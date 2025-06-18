import os
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal
from agent.model_adapter import MODEL_SERIES

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: Literal[
        "gemini-2.0-flash",
        "qwen-turbo",
        "qwen-plus",
        "qwen-max"
    ] = Field(
        default="gemini-2.0-flash" if MODEL_SERIES == "gemini" else "qwen-plus",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reasoning_model: Literal[
        "gemini-2.0-flash",
        "qwen-turbo",
        "qwen-plus",
        "qwen-max"
    ] = Field(
        default="gemini-2.0-flash" if MODEL_SERIES == "gemini" else "qwen-max",
        metadata={
            "description": "The name of the language model to use for the agent's reasoning."
        },
    )

    answer_model: Literal[
        "gemini-2.5-pro-preview-05-06",
        "qwen-turbo",
        "qwen-plus",
        "qwen-max"
    ] = Field(
        default="gemini-2.5-pro-preview-05-06" if MODEL_SERIES == "gemini" else "qwen-plus",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={
            "description": "The number of initial search queries to generate."
        },
    )

    max_research_loops: int = Field(
        default=3,
        metadata={
            "description": "The maximum number of research loops to perform."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: dict) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {})
        return cls(**configurable)
