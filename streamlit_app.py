# ============================================================
# Supply Chain Control Tower — Multi‑Agent Streamlit App
# ============================================================

import os
import asyncio
import streamlit as st

from langchain_openai import ChatOpenAI
from orxhestra import LlmAgent, Runner, InMemorySessionService
from orxhestra.tools.agent_tool import AgentTool



# ------------------------------------------------------------
# Helper: consume async generator safely in Streamlit
# ------------------------------------------------------------
def run_async_generator(async_gen):
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
# Streamlit config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Control Tower — Multi‑Agent Demo",
    layout="wide",
)

st.title("🛰️ Supply Chain Control Tower — Multi‑Agent Demo")


# ------------------------------------------------------------
# Sidebar (controls + trace)
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", ["gpt-4o-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.subheader("🔍 Live Delegation Trace")
    trace_box = st.empty()
    st.caption("Shows when the manager calls specialist agents")


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "buffers" not in st.session_state:
    st.session_state.buffers = {}

if "trace" not in st.session_state:
    st.session_state.trace = []


# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
llm = ChatOpenAI(model=model_name, temperature=temperature)


# ------------------------------------------------------------
# Specialist agents
# ------------------------------------------------------------
inventory_projection_agent = LlmAgent(
    name="InventoryProjectionAgent",
    llm=llm,
    instructions=(
        "You are an Inventory Projection Analyst.\n"
        "Explain inventory projection and KPI impact (Service Level, DOI).\n"
        "Structure:\n"
        "- Executive bullets\n"
        "- KPI impact\n"
        "- Assumptions / scenario levers\n"
        "- Recommended actions"
    ),
)

live_factpack_agent = LlmAgent(
    name="LiveFactpackAgent",
    llm=llm,
    instructions=(
        "You are a Live Factpack Analyst.\n"
        "Summarize current performance and exceptions.\n"
        "Structure:\n"
        "- Headline performance\n"
        "- Top exceptions\n"
        "- Likely drivers\n"
        "- Next actions"
    ),
)

stockout_rootcause_agent = LlmAgent(
    name="StockoutRootCauseAgent",
    llm=llm,
    instructions=(
        "You are a Stockout / Availability Analyst.\n"
        "Explain why availability dropped.\n"
        "Structure:\n"
        "- What happened\n"
        "- Root causes\n"
        "- What to check\n"
        "- Corrective actions"
    ),
)


# ------------------------------------------------------------
# Manager agent (IMPORTANT: forces multi‑agent interaction)
# ------------------------------------------------------------
MANAGER_NAME = "SupplyTowerManager"

manager = LlmAgent(
    name=MANAGER_NAME,
    llm=llm,
    tools=[
        AgentTool(inventory_projection_agent),
        AgentTool(live_factpack_agent),
        AgentTool(stockout_rootcause_agent),
    ],
    instructions=(
        "You are the Supply Chain Control Tower Manager.\n\n"
        "For EVERY user request:\n"
        "1) Call ONE primary specialist based on intent.\n"
        "2) Call ONE secondary specialist to cross‑check.\n"
        "   - If primary = InventoryProjectionAgent → also call LiveFactpackAgent\n"
        "   - If primary = LiveFactpackAgent → also call StockoutRootCauseAgent\n"
        "   - If primary = StockoutRootCauseAgent → also call LiveFactpackAgent\n"
        "3) Synthesize a clear executive summary and actions.\n\n"
        "Routing hints:\n"
        "- projection / forecast / scenario / DOI → InventoryProjectionAgent\n"
        "- today / performance / factpack → LiveFactpackAgent\n"
        "- stockout / availability / service drop → StockoutRootCauseAgent"
    ),
)


runner = Runner(
    agent=manager,
    app_name="supply-chain-control-tower",
    session_service=InMemorySessionService(),
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with right:
    st.subheader("🧭 Control Tower Views")

    b1 = st.button("📌 Daily Live Factpack (exceptions)")
    b2 = st.button("📈 Inventory Projection (scenario impact)")
    b3 = st.button("🧯 Stockout RCA (why availability dropped)")

    st.divider()
    st.subheader("📂 Output (by agent)")
    output_placeholder = st.empty()


with left:
    st.subheader("💬 Control Tower Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(
        "Ask the Control Tower… (e.g. Why did service level drop this week?)"
    )


# ------------------------------------------------------------
# Map buttons → prompts
# ------------------------------------------------------------
if b1:
    user_prompt = "Give me today's live factpack with exceptions and next actions."
if b2:
    user_prompt = (
        "Project inventory and KPIs under baseline vs improved replenishment."
    )
if b3:
    user_prompt = (
        "Why did availability drop and what are the main stockout root causes?"
    )


# ------------------------------------------------------------
# Run agents
# ------------------------------------------------------------
if user_prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with left:
        with st.chat_message("user"):
            st.markdown(user_prompt)

    # Reset state per run
    st.session_state.buffers = {}
    st.session_state.trace = []
    trace_box.markdown("")

    with left:
        with st.chat_message("assistant"):
            assistant_placeholder = st.empty()

            async_gen = runner.astream(
                user_id="user1",
                session_id="s1",
                new_message=user_prompt,
            )

            for event in run_async_generator(async_gen):
                # IMPORTANT: use branch first for sub‑agents
                agent = (
                    getattr(event, "branch", None)
                    or getattr(event, "agent_name", None)
                    or "unknown"
                )
                text = getattr(event, "text", None)

                # ---- Trace delegation ----
                if getattr(event, "has_tool_calls", False):
                    tc = event.tool_calls[0]
                    st.session_state.trace.append(
                        f"**{MANAGER_NAME} → {tc.tool_name}**"
                    )
                    trace_box.markdown("\n".join(st.session_state.trace))

                # ---- Buffer per agent ----
                if text:
                    st.session_state.buffers.setdefault(agent, "")
                    st.session_state.buffers[agent] += text

                # ---- LEFT: manager only ----
                manager_text = st.session_state.buffers.get(
                    MANAGER_NAME, ""
                ).strip()
                if manager_text:
                    assistant_placeholder.markdown(manager_text)

                # ---- RIGHT: specialists only ----
                rendered = ""
                for a in [
                    "InventoryProjectionAgent",
                    "LiveFactpackAgent",
                    "StockoutRootCauseAgent",
                ]:
                    if a in st.session_state.buffers and st.session_state.buffers[a].strip():
                        rendered += f"### {a}\n{st.session_state.buffers[a].strip()}\n\n"

                output_placeholder.markdown(rendered)

            # Persist final manager answer
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": manager_text or "Done.",
                }
            )
