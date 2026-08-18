"""Day 26 - minimal Streamlit UI for the Employee Onboarding AI Agent.

The UI talks to the FastAPI backend over HTTP only; it never imports the agent
or RAG code directly. Start the backend first, then run:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Employee Onboarding Assistant", page_icon="🤝")
st.title("🤝 Employee Onboarding Assistant")
st.caption(f"Backend: {BACKEND_URL}  (synthetic data, educational demo)")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, response_dict)

with st.sidebar:
    st.header("Session")
    employee_id_raw = st.text_input("Employee ID (optional)", value="101")
    if st.button("Check backend health"):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            st.success("Backend OK" if r.ok else f"Backend error: {r.status_code}")
        except requests.RequestException:
            st.error("Backend unavailable.")

question = st.chat_input("Ask about company policies, or request an HR ticket...")

if question:
    employee_id = None
    if employee_id_raw.strip():
        try:
            employee_id = int(employee_id_raw.strip())
        except ValueError:
            st.warning("Employee ID must be a number; sending without it.")

    payload = {"message": question}
    if employee_id is not None:
        payload["employee_id"] = employee_id

    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
    except requests.Timeout:
        st.error("The request timed out. Please try again.")
        resp = None
    except requests.RequestException:
        st.error("Could not reach the backend. Is FastAPI running?")
        resp = None

    if resp is not None:
        if resp.ok:
            st.session_state.history.append((question, resp.json()))
        elif resp.status_code == 422:
            st.warning("Invalid input. Please check your message.")
        else:
            st.error(f"Backend error ({resp.status_code}).")

# Render chat history (oldest first).
for q, data in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(data.get("answer", ""))
        sources = data.get("sources") or []
        if sources:
            st.caption("Sources: " + ", ".join(sources))
        tools_used = data.get("tools_used") or []
        if tools_used:
            st.caption("Tools used: " + ", ".join(tools_used))
        if data.get("ticket_id"):
            st.success(f"HR ticket created: #{data['ticket_id']}")
