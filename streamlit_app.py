# ============================================================
# Supply Chain Control Tower — Step‑Based Orchestration View
# ============================================================

import os
import asyncio
import streamlit as st

from langchain_openai import ChatOpenAI
from orxhestra import LlmAgent, Runner, InMemorySessionService
from orxhestra.tools.agent_tool import AgentTool



# ------------------------------------------------------------
# Helper: run async generator in Streamlit
# ------------------------------------------------------------
def run_async(async_gen):
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Control Tower — Execution View",
    layout="wide",
)

st.title("🛰️ Supply Chain Control Tower — Execution View")


# ------------------------------------------------------------
# Sidebar: model config
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", ["gpt-4o-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "steps" not in st.session_state:
    st.session_state.steps = []

if "current_step" not in st.session_state:
    st.session_state.current_step = None


# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
llm = ChatOpenAI(model=model_name, temperature=temperature)


# ------------------------------------------------------------
# Specialist agents
# ------------------------------------------------------------
inventory_agent = LlmAgent(
    name="InventoryProjectionAgent",
    llm=llm,
    instructions="Provide inventory projection and KPI implications."
)

factpack_agent = LlmAgent(
    name="LiveFactpackAgent",
    llm=llm,
    instructions="Provide current performance and key exceptions."
)


# ------------------------------------------------------------
# Manager with EXPLICIT STEP CONTRACT
# ------------------------------------------------------------
MANAGER_NAME = "SupplyTowerManager"

manager = LlmAgent(
    name=MANAGER_NAME,
    llm=llm,
    tools=[
        AgentTool(inventory_agent),
        AgentTool(factpack_agent),
    ],
    instructions=(
        "You are the Supply Chain Control Tower Manager.\n\n"
        "You MUST follow these steps in order:\n"
        "STEP 1: Call InventoryProjectionAgent to understand forward-looking impact.\n"
        "STEP 2: Call LiveFactpackAgent to validate against current performance.\n"
        "STEP 3: Synthesize an executive summary with decisions and actions.\n\n"
        "Do not skip steps."
    ),
    max_iterations=3,
)


runner = Runner(
    agent=manager,
    app_name="control-tower-execution",
    session_service=InMemorySessionService(),
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")


# ------------------------------------------------------------
# Right panel: EXECUTION TRACKER
# ------------------------------------------------------------
with right:
    st.subheader("🧭 Execution Steps")
    steps_box = st.container()


# ------------------------------------------------------------
# Left panel: chat
# ------------------------------------------------------------
with left:
    st.subheader("💬 Control Tower Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(
        "Ask the Control Tower (e.g. Why did service level drop this week?)"
    )


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if user_prompt:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with left:
        with st.chat_message("user"):
            st.markdown(user_prompt)

    # Reset steps
    st.session_state.steps = [
        {"name": "Inventory projection", "status": "pending"},
        {"name": "Validate with live performance", "status": "pending"},
        {"name": "Executive synthesis & actions", "status": "pending"},
    ]

    def render_steps():
        steps_box.empty()
        with steps_box:
            for s in st.session_state.steps:
                icon = {
                    "pending": "⏳",
                    "running": "▶️",
                    "done": "✅",
                }[s["status"]]
                st.markdown(f"{icon} **{s['name']}**")

    render_steps()

    with left:
        with st.chat_message("assistant"):
            assistant_box = st.empty()

            async_gen = runner.astream(
                user_id="user1",
                session_id="s1",
                new_message=user_prompt,
            )

            for event in run_async(async_gen):
                # Detect tool calls → map to steps
                if getattr(event, "has_tool_calls", False):
                    tc = event.tool_calls[0]

                    if tc.tool_name == "InventoryProjectionAgent":
                        st.session_state.steps[0]["status"] = "running"
                        render_steps()
                        st.session_state.steps[0]["status"] = "done"
                        st.session_state.steps[1]["status"] = "running"
                        render_steps()

                    if tc.tool_name == "LiveFactpackAgent":
                        st.session_state.steps[1]["status"] = "done"
                        st.session_state.steps[2]["status"] = "running"
                        render_steps()

                # Final synthesis
                if getattr(event, "agent_name", None) == MANAGER_NAME:
                    if getattr(event, "text", None):
                        assistant_box.markdown(event.text)

                if hasattr(event, "is_final_response") and event.is_final_response():
                    st.session_state.steps[2]["status"] = "done"
                    render_steps()

            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_box}
            )
