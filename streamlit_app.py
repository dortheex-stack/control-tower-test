import streamlit as st
import time
import asyncio

from langchain_openai import ChatOpenAI
from orxhestra import LlmAgent, Runner, InMemorySessionService
from orxhestra.tools.agent_tool import AgentTool


# -----------------------
# Helpers: run async stream safely inside Streamlit (sync loop)
# -----------------------
def run_async_generator(async_gen):
    """
    Consume an async generator from a sync Streamlit script.
    Uses a dedicated event loop to avoid 'event loop already running' issues.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                item = loop.run_until_complete(async_gen.__anext__())
                yield item
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(page_title="Supply Chain Control Tower (Agents)", layout="wide")
st.title("🛰️ Supply Chain Control Tower — Multi‑Agent Demo")


# -----------------------
# Sidebar: configuration & trace
# -----------------------
with st.sidebar:
    st.header("⚙️ Control Tower Setup")
    model_name = st.selectbox("Model", ["gpt-4o-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    st.divider()
    st.subheader("🔍 Live Trace (Delegation)")
    trace_box = st.empty()
    st.caption("Shows when the manager routes work to specialist agents.")


# -----------------------
# Session state
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trace" not in st.session_state:
    st.session_state.trace = []
if "buffers" not in st.session_state:
    st.session_state.buffers = {}


# -----------------------
# LLM
# -----------------------
llm = ChatOpenAI(model=model_name, temperature=temperature)


# -----------------------
# Specialist Agents (Supply Chain Control Tower roles)
# -----------------------

inventory_projection_agent = LlmAgent(
    name="InventoryProjectionAgent",
    llm=llm,
    description="Explains inventory projection over time, KPI projection (e.g., Service Level, DOI), and scenario impacts.",
    instructions=(
        "You are the Inventory Projection Analyst in a supply chain control tower.\n"
        "Focus on: inventory projection over time, KPI projection (Service Level, DOI), and scenario-based impacts.\n"
        "Output format:\n"
        "1) Executive summary (3 bullets)\n"
        "2) KPI impacts (Service Level, DOI) as bullets\n"
        "3) Key assumptions & scenario levers (bullets)\n"
        "4) Recommended actions (bullets)\n"
        "Keep it practical and decision-oriented."
    ),
)

live_factpack_agent = LlmAgent(
    name="LiveFactpackAgent",
    llm=llm,
    description="Summarizes current performance and highlights exceptions for drill-down (e.g., plant/site/supply group).",
    instructions=(
        "You are the Live Factpack Analyst in a supply chain control tower.\n"
        "Focus on: current performance snapshot, exceptions, and what to drill into.\n"
        "Output format:\n"
        "1) 'Here & now' performance headline (max 2 lines)\n"
        "2) Top 5 exceptions (bullet list)\n"
        "3) Likely drivers (bullets)\n"
        "4) Next actions / who should act (bullets)\n"
        "Be concise and action-oriented."
    ),
)

stockout_rootcause_agent = LlmAgent(
    name="StockoutRootCauseAgent",
    llm=llm,
    description="Explains likely root causes behind stockouts/availability drops and suggests corrective actions.",
    instructions=(
        "You are the Availability / Stockout Root Cause Analyst.\n"
        "Your job is to explain why service/availability dropped and what the biggest drivers of stockouts are.\n"
        "Use a practical operations lens.\n"
        "Output format:\n"
        "1) What happened (2 bullets)\n"
        "2) Top root causes (bullets)\n"
        "3) What to check next (bullets)\n"
        "4) Suggested corrective actions (bullets)\n"
        "If data is missing, state what you would need to confirm."
    ),
)

# -----------------------
# Manager Agent (routes to specialists via AgentTool)
# -----------------------
control_tower_manager = LlmAgent(
    name="SupplyTowerManager",
    llm=llm,
    tools=[
        AgentTool(inventory_projection_agent),
        AgentTool(live_factpack_agent),
        AgentTool(stockout_rootcause_agent),
    ],
    instructions=(
        "You are the Supply Chain Control Tower Manager.\n"
        "Your task is to route the user's request to the right specialist agents and synthesize an executive answer.\n\n"
        "Routing rules:\n"
        "- If the user asks about 'projection', 'forecast', 'scenario', 'DOI' → call InventoryProjectionAgent.\n"
        "- If the user asks about 'today', 'current performance', 'factpack', 'exceptions' → call LiveFactpackAgent.\n"
        "- If the user asks about 'stockout', 'availability', 'service level drop', 'root cause' → call StockoutRootCauseAgent.\n\n"
        "Always:\n"
        "1) Delegate to the relevant agent(s)\n"
        "2) Summarize into a crisp control-tower style update\n"
        "3) End with recommended actions / decisions."
    ),
)

runner = Runner(
    agent=control_tower_manager,
    app_name="supply-chain-control-tower",
    session_service=InMemorySessionService(),
)


# -----------------------
# Main UI: quick-start buttons + chat
# -----------------------
colA, colB = st.columns([3, 2], gap="large")

with colB:
    st.subheader("🧭 Control Tower Views")
    st.caption("Quick prompts to simulate typical control tower use cases.")
    b1 = st.button("📌 Daily Live Factpack (exceptions)")
    b2 = st.button("📈 Inventory Projection (scenario impact)")
    b3 = st.button("🧯 Stockout RCA (why availability dropped)")

    st.divider()
    st.subheader("🗂️ Output (by agent)")
    output_placeholder = st.empty()

with colA:
    st.subheader("💬 Control Tower Assistant")
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask the Control Tower… (e.g., 'Why did service level drop this week?')")


# Map buttons to prompts
if b1:
    user_prompt = "Give me today's live factpack: headline performance + top 5 exceptions + next actions."
if b2:
    user_prompt = "Project inventory and KPIs (Service Level, DOI) under two scenarios: baseline vs improved replenishment."
if b3:
    user_prompt = "Why did availability drop and what are the biggest root causes of stockouts? Suggest corrective actions."


# -----------------------
# Execute run if user_prompt exists
# -----------------------
if user_prompt:
    # Append user message to chat
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with colA:
        with st.chat_message("user"):
            st.markdown(user_prompt)

    # Reset per-run buffers + trace
    st.session_state.buffers = {}
    st.session_state.trace = []
    trace_box.markdown("")

    # Create assistant chat bubble
    with colA:
        with st.chat_message("assistant"):
            assistant_placeholder = st.empty()

            # Run the async event stream, but consume synchronously for Streamlit
            async_gen = runner.astream(
                user_id="user1",
                session_id="s1",
                new_message=user_prompt,
            )

            final_manager_answer = ""

            for event in run_async_generator(async_gen):
                agent = getattr(event, "agent_name", None) or getattr(event, "branch", None)
                text = getattr(event, "text", None)

                # Live trace: manager delegation (tool calls)
                if getattr(event, "has_tool_calls", False):
                    tc = event.tool_calls[0]
                    st.session_state.trace.append(f"**SupplyTowerManager → {tc.tool_name}**")
                    trace_box.markdown("\n".join(st.session_state.trace))

                # Buffer streaming text per agent for readable output
                if agent and text:
                    st.session_state.buffers.setdefault(agent, "")
                    st.session_state.buffers[agent] += text

                    # Update right-hand "by agent" output live
                    rendered = ""
                    for a, content in st.session_state.buffers.items():
                        rendered += f"### {a}\n{content.strip()}\n\n"
                    output_placeholder.markdown(rendered)

                    # Update main assistant bubble with manager buffer if available
                    # (Manager may stream too; we show the latest manager buffer as the "main" answer)
                    if a == "SupplyTowerManager":
                        final_manager_answer = st.session_state.buffers.get("SupplyTowerManager", "")

                # When an event marks final response, keep the final manager answer clean
                if hasattr(event, "is_final_response") and event.is_final_response():
                    # Prefer manager final text if present
                    if agent == "SupplyTowerManager" and getattr(event, "text", None):
                        final_manager_answer = st.session_state.buffers.get("SupplyTowerManager", "").strip()

                # Keep the main bubble updated (even if manager doesn't stream much)
                if final_manager_answer:
                    assistant_placeholder.markdown(final_manager_answer)

            # Persist assistant final to chat history
            st.session_state.messages.append({"role": "assistant", "content": final_manager_answer or "Done."})
