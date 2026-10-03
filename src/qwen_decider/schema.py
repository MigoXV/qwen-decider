from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1, max_length=16384)
    criteria: dict[str, str] = Field(min_length=2, max_length=10)

    @field_validator("instructions")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def keys_nonempty(cls, value):
        if any(not k.strip() for k in value):
            raise ValueError("option keys must not be blank")
        return value


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1, max_length=8)
