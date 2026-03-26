# ============================================================
# Supply Chain Control Tower — Command Center UI
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
    page_title="Supply Chain Control Tower",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Dark command-center CSS
# ------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Global reset ── */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #080C12 !important;
    color: #C8D6E5 !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  [data-testid="stSidebar"] {
    background-color: #0D1117 !important;
    border-right: 1px solid #1C2A3A !important;
  }

  /* ── Header ── */
  .ct-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 0 24px 0;
    border-bottom: 1px solid #1C2A3A;
    margin-bottom: 24px;
  }
  .ct-header .pulse-dot {
    width: 10px; height: 10px;
    background: #00E5A0;
    border-radius: 50%;
    box-shadow: 0 0 8px #00E5A0;
    animation: pulse 2s infinite;
    flex-shrink: 0;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #00E5A0; }
    50%       { opacity: 0.4; box-shadow: 0 0 2px #00E5A0; }
  }
  .ct-header h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #E8F4FF !important;
    margin: 0 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .ct-header .sub {
    font-size: 0.72rem;
    color: #4A6A80;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
  }

  /* ── Step tracker ── */
  .step-card {
    background: #0D1117;
    border: 1px solid #1C2A3A;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: border-color 0.3s, box-shadow 0.3s;
    position: relative;
    overflow: hidden;
  }
  .step-card.running {
    border-color: #0077FF;
    box-shadow: 0 0 18px rgba(0,119,255,0.15);
  }
  .step-card.done {
    border-color: #00E5A0;
    box-shadow: 0 0 10px rgba(0,229,160,0.08);
  }
  .step-card.pending {
    opacity: 0.5;
  }
  .step-card .progress-bar {
    position: absolute;
    bottom: 0; left: 0;
    height: 2px;
    background: linear-gradient(90deg, #0077FF, #00E5A0);
    animation: scan 1.2s ease-in-out infinite;
  }
  @keyframes scan {
    0%   { width: 0%;   opacity: 1; }
    70%  { width: 100%; opacity: 1; }
    100% { width: 100%; opacity: 0; }
  }
  .step-icon {
    font-size: 1.1rem;
    width: 28px;
    text-align: center;
    flex-shrink: 0;
  }
  .step-info { flex: 1; min-width: 0; }
  .step-name {
    font-size: 0.82rem;
    font-weight: 500;
    color: #C8D6E5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .agent-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    padding: 2px 7px;
    border-radius: 3px;
    margin-top: 4px;
    letter-spacing: 0.06em;
    font-weight: 700;
  }
  .badge-inventory { background: #0A1F3A; color: #3399FF; border: 1px solid #1A3A6A; }
  .badge-factpack  { background: #0A2A1A; color: #00E5A0; border: 1px solid #1A5A3A; }
  .badge-manager   { background: #1A1020; color: #B080FF; border: 1px solid #3A1A6A; }
  .step-status {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    flex-shrink: 0;
  }
  .status-pending { color: #3A5A70; }
  .status-running { color: #0077FF; }
  .status-done    { color: #00E5A0; }

  /* ── Section label ── */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #2A4A60;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1C2A3A;
  }

  /* ── Sidebar quick actions ── */
  .sidebar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #2A4A60;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1C2A3A;
  }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
    background: #0D1117 !important;
    border: 1px solid #1C2A3A !important;
    border-radius: 8px !important;
    margin-bottom: 10px !important;
    padding: 12px 16px !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: #1A3A6A !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-color: #1C2A3A !important;
  }

  /* ── Chat input ── */
  [data-testid="stChatInput"] textarea {
    background: #0D1117 !important;
    border: 1px solid #1C2A3A !important;
    border-radius: 8px !important;
    color: #C8D6E5 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: #0077FF !important;
    box-shadow: 0 0 12px rgba(0,119,255,0.15) !important;
  }

  /* ── Streamlit selectbox / slider ── */
  [data-testid="stSelectbox"] label,
  [data-testid="stSlider"] label {
    color: #4A6A80 !important;
    font-size: 0.75rem !important;
    font-family: 'Space Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Quick-action buttons ── */
  .stButton > button {
    background: #0D1117 !important;
    border: 1px solid #1C2A3A !important;
    color: #6A9AB0 !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    width: 100% !important;
    text-align: left !important;
    padding: 10px 14px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    border-color: #0077FF !important;
    color: #C8D6E5 !important;
    box-shadow: 0 0 14px rgba(0,119,255,0.12) !important;
    background: #0A1525 !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
  [data-testid="collapsedControl"] { color: #4A6A80 !important; }

  /* ── Divider ── */
  hr { border-color: #1C2A3A !important; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "steps" not in st.session_state:
    st.session_state.steps = []
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 8px 0;">
      <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                  text-transform:uppercase;letter-spacing:0.12em;color:#2A4A60;">
        Control Tower
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:1.0rem;
                  font-weight:700;color:#E8F4FF;margin-top:2px;">
        ORXHESTRA
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("", ["gpt-4o-mini", "gpt-4o"], index=0, label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()

    st.markdown('<div class="sidebar-label">Quick Actions</div>', unsafe_allow_html=True)

    if st.button("📦  Inventory Report"):
        st.session_state.quick_prompt = "Give me a full inventory projection report including KPI implications and recommended actions."

    if st.button("📉  Service Level Drop"):
        st.session_state.quick_prompt = "Why did our service level drop this week? Provide root cause analysis and corrective actions."

    if st.button("🚚  Delivery Performance"):
        st.session_state.quick_prompt = "Analyze delivery performance trends and highlight key exceptions requiring immediate attention."

    st.divider()

    if st.button("🗑  Clear Session"):
        st.session_state.messages = []
        st.session_state.steps = []
        st.rerun()

    st.markdown("""
    <div style="position:fixed;bottom:20px;left:0;width:240px;
                padding:0 20px;box-sizing:border-box;">
      <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                  color:#1C3A50;letter-spacing:0.08em;line-height:1.8;">
        SUPPLY CHAIN CONTROL TOWER<br>
        POWERED BY ORXHESTRA
      </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# LLM + Agents
# ------------------------------------------------------------
llm = ChatOpenAI(model=model_name, temperature=temperature)

inventory_agent = LlmAgent(
    name="InventoryProjectionAgent",
    llm=llm,
    instructions="Provide inventory projection and KPI implications.",
)

factpack_agent = LlmAgent(
    name="LiveFactpackAgent",
    llm=llm,
    instructions="Provide current performance and key exceptions.",
)

MANAGER_NAME = "SupplyTowerManager"
manager = LlmAgent(
    name=MANAGER_NAME,
    llm=llm,
    tools=[AgentTool(inventory_agent), AgentTool(factpack_agent)],
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
    app_name="control-tower",
    session_service=InMemorySessionService(),
)


# ------------------------------------------------------------
# Step renderer helper
# ------------------------------------------------------------
STEP_DEFS = [
    {
        "name":   "Inventory Projection",
        "agent":  "InventoryProjectionAgent",
        "badge":  "badge-inventory",
        "label":  "INV · AGENT",
    },
    {
        "name":   "Live Performance Validation",
        "agent":  "LiveFactpackAgent",
        "badge":  "badge-factpack",
        "label":  "FACTPACK · AGENT",
    },
    {
        "name":   "Executive Synthesis",
        "agent":  "SupplyTowerManager",
        "badge":  "badge-manager",
        "label":  "MANAGER · AGENT",
    },
]

def render_steps(steps_placeholder):
    cards_html = ""
    for i, step in enumerate(st.session_state.steps):
        status   = step["status"]
        defn     = STEP_DEFS[i]

        status_icon = {"pending": "○", "running": "◎", "done": "●"}[status]
        status_cls  = f"status-{status}"
        card_cls    = f"step-card {status}"
        bar_html    = '<div class="progress-bar"></div>' if status == "running" else ""

        cards_html += f"""
        <div class="{card_cls}">
          {bar_html}
          <div class="step-icon">{status_icon}</div>
          <div class="step-info">
            <div class="step-name">{defn['name']}</div>
            <span class="agent-badge {defn['badge']}">{defn['label']}</span>
          </div>
          <div class="step-status {status_cls}">{status.upper()}</div>
        </div>
        """

    steps_placeholder.markdown(cards_html, unsafe_allow_html=True)


# ------------------------------------------------------------
# Main layout
# ------------------------------------------------------------
st.markdown("""
<div class="ct-header">
  <div class="pulse-dot"></div>
  <div>
    <h1>Supply Chain Control Tower</h1>
    <div class="sub">Live Orchestration · Multi-Agent</div>
  </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with right:
    st.markdown('<div class="section-label">Execution Pipeline</div>', unsafe_allow_html=True)
    steps_placeholder = st.empty()

    # Render initial steps if present, else idle state
    if st.session_state.steps:
        render_steps(steps_placeholder)
    else:
        idle_html = ""
        for defn in STEP_DEFS:
            idle_html += f"""
            <div class="step-card pending">
              <div class="step-icon">○</div>
              <div class="step-info">
                <div class="step-name">{defn['name']}</div>
                <span class="agent-badge {defn['badge']}">{defn['label']}</span>
              </div>
              <div class="step-status status-pending">IDLE</div>
            </div>
            """
        steps_placeholder.markdown(idle_html, unsafe_allow_html=True)

with left:
    st.markdown('<div class="section-label">Assistant</div>', unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Resolve prompt: quick action takes priority
    user_prompt = st.chat_input("Ask the Control Tower…")
    if st.session_state.quick_prompt:
        user_prompt = st.session_state.quick_prompt
        st.session_state.quick_prompt = None


# ------------------------------------------------------------
# Run on prompt
# ------------------------------------------------------------
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Initialize steps
    st.session_state.steps = [
        {"name": s["name"], "status": "pending"} for s in STEP_DEFS
    ]
    render_steps(steps_placeholder)

    with left:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            assistant_box = st.empty()

            async_gen = runner.astream(
                user_id="user1",
                session_id="s1",
                new_message=user_prompt,
            )

            for event in run_async(async_gen):
                if getattr(event, "has_tool_calls", False):
                    tc = event.tool_calls[0]

                    if tc.tool_name == "InventoryProjectionAgent":
                        st.session_state.steps[0]["status"] = "running"
                        render_steps(steps_placeholder)
                        st.session_state.steps[0]["status"] = "done"
                        st.session_state.steps[1]["status"] = "running"
                        render_steps(steps_placeholder)

                    if tc.tool_name == "LiveFactpackAgent":
                        st.session_state.steps[1]["status"] = "done"
                        st.session_state.steps[2]["status"] = "running"
                        render_steps(steps_placeholder)

                if getattr(event, "agent_name", None) == MANAGER_NAME:
                    if getattr(event, "text", None):
                        assistant_box.markdown(event.text)

                if hasattr(event, "is_final_response") and event.is_final_response():
                    st.session_state.steps[2]["status"] = "done"
                    render_steps(steps_placeholder)

            final_text = assistant_box._value if hasattr(assistant_box, "_value") else ""
            st.session_state.messages.append(
                {"role": "assistant", "content": final_text}
            )
