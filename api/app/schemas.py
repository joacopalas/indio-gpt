from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    candidate: str = "bpe500"

    prompt: str = "<|song|>\n"

    max_new_tokens: int = Field(default=150, ge=1, le=300)
    temperature: float = Field(default=0.85, ge=0.01, le=2.0)
    top_k: int = Field(default=40, ge=1, le=200)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    rep_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    rep_window: int = Field(default=64, ge=0, le=256)

    slide_keep: Optional[int] = Field(default=None, ge=8, le=128)

    stop_at_end: bool = True
    include_specials: bool = False
    stream: bool = False

    seed: Optional[int] = None