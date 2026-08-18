"""Day 23 - FastAPI routes for the Employee Onboarding AI Agent.

Endpoints:
    GET  /health              -> service health
    POST /documents/ingest    -> (re)build the Chroma index from company_docs
    POST /chat                -> run the onboarding agent
    GET  /employees/{id}      -> fictional employee profile (404 if unknown)
    GET  /tickets             -> all HR tickets from SQLite

Errors are translated into safe HTTP responses; Python stack traces are never
returned to clients. Secrets never appear in any response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

import model_client
from app import employees
from app.agent.onboarding_agent import run_agent
from app.database import tickets
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    EmployeeProfile,
    HealthResponse,
    IngestResponse,
    Ticket,
)
from app.rag.ingestion import ingest_documents

logger = logging.getLogger("api")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/documents/ingest", response_model=IngestResponse)
def documents_ingest() -> IngestResponse:
    try:
        result = ingest_documents()
    except model_client.ConfigurationError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Service is not configured correctly.")
    except model_client.ModelClientError:
        logger.exception("Ingestion failed: provider error")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            detail="Document ingestion failed. Please try again.")
    return IngestResponse(
        status="success",
        documents=result["documents"],
        chunks=result["chunks"],
        collection=result["collection"],
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info("Chat request (has_employee=%s)", request.employee_id is not None)
    try:
        result = run_agent(request.message, employee_id=request.employee_id)
    except model_client.ConfigurationError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Service is not configured correctly.")
    except model_client.AuthenticationError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Upstream service is unavailable.")
    except model_client.ModelClientError:
        logger.exception("Chat failed: provider error")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            detail="The assistant is temporarily unavailable.")
    except Exception:  # noqa: BLE001 - last-resort guard, no stack traces to client
        logger.exception("Chat failed: unexpected error")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred.")

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        tools_used=result["tools_used"],
        ticket_id=result["ticket_id"],
    )


@router.get("/employees/{employee_id}", response_model=EmployeeProfile)
def get_employee(employee_id: int) -> EmployeeProfile:
    try:
        profile = employees.get_employee(employee_id)
    except employees.EmployeeNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail=f"Employee {employee_id} not found.")
    return EmployeeProfile(**profile)


@router.get("/tickets", response_model=list[Ticket])
def list_tickets() -> list[Ticket]:
    return [Ticket(**t) for t in tickets.get_all_tickets()]
