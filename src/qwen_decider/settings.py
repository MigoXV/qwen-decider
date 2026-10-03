from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_path: Path
    precision: Literal["bf16", "fp16", "int8"] = "bf16"
    max_input_tokens: int = Field(default=2048, ge=1, le=2048)
    max_num_seqs: int = Field(default=1, ge=1, le=1)
    max_requests: int = Field(default=16, ge=1, le=128)
    request_timeout: float = Field(default=30, gt=0, le=300)
    gpu_memory_utilization: float = Field(default=0.12, gt=0, lt=1)
    enforce_eager: bool = False

    kv_cache_memory_bytes: int = Field(default=128 * 1024**2, ge=64 * 1024**2, le=128 * 1024**2)
    max_num_batched_tokens: int = Field(default=2048, ge=256, le=2048)
    gpu_memory_limit_bytes: int = Field(default=8_000_000_000, ge=6_000_000_000, le=8_000_000_000)
