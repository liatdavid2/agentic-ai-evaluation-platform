from pydantic import BaseModel, Field
from typing import Literal


class BenchmarkRequest(BaseModel):
    task_count: int = Field(default=20, ge=1, le=500)
    repeats: int = Field(default=3, ge=1, le=10)

    configuration: Literal[
        "tools",
        "guardrailed",
    ] = "guardrailed"

    provider: Literal[
        "heuristic",
        "openai",
    ] = "openai"
