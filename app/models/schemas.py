"""Pydantic schemas shared by the API and the agent.

These describe the *public* request/response shapes. Internal debugging data
(raw retrieved chunks, distances, chain internals) is deliberately kept OUT of
these models so it never leaks to API clients.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Health / ingestion
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = "ok"


class IngestResponse(BaseModel):
    status: str = "success"
    documents: int
    chunks: int
    collection: str


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    employee_id: int | None = Field(
        default=None,
        description="Optional employee id for profile-aware answers.",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty or whitespace-only.")
        return value


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    ticket_id: int | None = None


# --------------------------------------------------------------------------- #
# Employees
# --------------------------------------------------------------------------- #
class EmployeeProfile(BaseModel):
    employee_id: int
    name: str
    department: str
    role: str
    country: str
    start_date: str


# --------------------------------------------------------------------------- #
# Tickets
# --------------------------------------------------------------------------- #
class Ticket(BaseModel):
    ticket_id: int
    employee_id: int
    topic: str
    description: str
    status: str
    created_at: str
