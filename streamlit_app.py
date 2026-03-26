# ============================================================
# Supply Chain Control Tower
# Danish Grocery Cooperative — Light Classic Theme
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
    page_title="Forsyningskæde · Control Tower",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# CSS — Light Classic / Scandinavian Editorial
# ------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Source+Sans+3:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  :root {
    --bg:             #F7F4F0;
    --surface:        #FFFFFF;
    --border:         #DDD8D0;
    --border-soft:    #EAE6E0;
    --text-primary:   #1A1612;
    --text-secondary: #5C5650;
    --text-muted:     #9C9590;
    --red:            #C0392B;
    --red-light:      #F9EDEB;
    --red-mid:        #E8B4AF;
    --blue:           #1A4A8A;
    --blue-light:     #EBF0F8;
    --green:          #1E6B4A;
    --green-light:    #EAF4EE;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Source Sans 3', sans-serif !important;
  }

  [data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  /* Sidebar brand */
  .sidebar-brand {
    padding: 20px 0 16px 0;
    border-bottom: 2px solid var(--red);
    margin-bottom: 20px;
  }
  .sidebar-brand .coop-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--red);
    font-weight: 500;
  }
  .sidebar-brand .brand-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    margin-top: 3px;
  }
  .sidebar-brand .brand-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text-muted);
    margin: 18px 0 8px 0;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border-soft);
  }

  /* Quick-action buttons */
  .stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    text-align: left !important;
    padding: 9px 12px !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
  }
  .stButton > button:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
    background: var(--red-light) !important;
  }

  [data-testid="stSelectbox"] label,
  [data-testid="stSlider"] label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* Page header */
  .ct-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 8px 0 18px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
  }
  .ct-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    line-height: 1.1;
  }
  .ct-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 3px;
    letter-spacing: 0.02em;
  }
  .live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--green-light);
    border: 1px solid #A8D5BC;
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    color: var(--green);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .live-dot {
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse-g 2s infinite;
  }
  @keyframes pulse-g {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  /* DC strip */
  .dc-strip {
    display: flex;
    gap: 8px;
    margin-bottom: 22px;
    flex-wrap: wrap;
  }
  .dc-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-primary);
    font-weight: 500;
    white-space: nowrap;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .dc-pill span {
    display: block;
    font-size: 0.54rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1px;
    font-weight: 400;
  }

  /* Section label */
  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text-muted);
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-soft);
  }

  /* Step cards */
  .step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 4px;
    padding: 13px 16px;
    margin-bottom: 9px;
    display: flex;
    align-items: center;
    gap: 14px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
  }
  .step-card.pending { opacity: 0.5; }
  .step-card.running {
    border-left-color: var(--blue);
    background: var(--blue-light);
    box-shadow: 0 2px 10px rgba(26,74,138,0.1);
  }
  .step-card.done {
    border-left-color: var(--green);
    background: var(--green-light);
  }
  .step-progress {
    position: absolute;
    bottom: 0; left: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--blue), #4A90D9);
    animation: sweep 1.4s ease-in-out infinite;
  }
  @keyframes sweep {
    0%   { width: 0;    opacity: 1; }
    65%  { width: 100%; opacity: 1; }
    100% { width: 100%; opacity: 0; }
  }
  .step-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    flex-shrink: 0;
  }
  .step-num.pending { background: #EAE6E0; color: var(--text-muted); }
  .step-num.running { background: var(--blue); color: #FFF; }
  .step-num.done    { background: var(--green); color: #FFF; }

  .step-info { flex: 1; min-width: 0; }
  .step-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .step-card.pending .step-name { color: var(--text-secondary); }

  .agent-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 3px;
    margin-top: 3px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-inv     { background: #E8EEF7; color: var(--blue);  border: 1px solid #C0CDE0; }
  .badge-fact    { background: var(--green-light); color: var(--green); border: 1px solid #A8D5BC; }
  .badge-manager { background: var(--red-light); color: var(--red); border: 1px solid var(--red-mid); }

  .step-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    flex-shrink: 0;
  }
  .status-pending { color: var(--text-muted); }
  .status-running { color: var(--blue); font-weight: 500; }
  .status-done    { color: var(--green); font-weight: 500; }

  /* Chat */
  [data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    margin-bottom: 10px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] span,
  [data-testid="stChatMessage"] div {
    color: var(--text-primary) !important;
  }
  [data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.88rem !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 3px rgba(192,57,43,0.08) !important;
    outline: none !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

  hr { border-color: var(--border-soft) !important; margin: 14px 0 !important; }
  #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
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
    <div class="sidebar-brand">
      <div class="coop-label">Dansk Andelsselskab · Supply Chain</div>
      <div class="brand-name">Control Tower</div>
      <div class="brand-sub">940 butikker · 6 distributionscentre</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("", ["gpt-4o-mini", "gpt-4o"], index=0, label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.markdown('<div class="sidebar-section">Hurtige analyser</div>', unsafe_allow_html=True)

    QUICK_ACTIONS = [
        ("📦", "Lagerrapport",
         "Give me a full inventory projection report across all 6 distribution centres "
         "(3 dry goods, 2 cold/fresh, 1 frozen). Include KPI implications, stock coverage "
         "days, and recommended replenishment actions."),
        ("📉", "Serviceniveaufald",
         "Why did our service level drop this week? Provide a root cause analysis across "
         "distribution centre types (dry, cold, frozen) and recommended corrective actions."),
        ("🚚", "Leveringsperformance",
         "Analyse delivery performance for the latest period across all 6 distribution centres. "
         "Highlight exceptions, on-time delivery rates by DC type, and top issues requiring "
         "immediate attention across our 940-store network."),
    ]

    for icon, label, prompt in QUICK_ACTIONS:
        if st.button(f"{icon}  {label}"):
            st.session_state.quick_prompt = prompt

    st.markdown("---")

    if st.button("↺  Ryd session"):
        st.session_state.messages = []
        st.session_state.steps = [{"status": "pending"} for _ in range(3)]
        st.rerun()

    st.markdown("""
    <div style="margin-top:28px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                  color:#C0BAB4;line-height:2.0;letter-spacing:0.06em;">
        TØRVARER · 3 DC&apos;ER<br>
        KØLEVARER · 2 DC&apos;ER<br>
        FROST · 1 DC<br>
        ~38.500 MEDARBEJDERE
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
    instructions=(
        "You are the Inventory Projection Specialist for a large Danish grocery cooperative "
        "with ~940 stores across four retail chains and 6 distribution centres "
        "(3 dry goods, 2 cold/fresh, 1 frozen). "
        "Provide forward-looking inventory projections, stock coverage analysis, "
        "and KPI implications. Be specific about DC types where relevant."
    ),
)

factpack_agent = LlmAgent(
    name="LiveFactpackAgent",
    llm=llm,
    instructions=(
        "You are the Live Performance Analyst for a large Danish grocery cooperative "
        "with ~940 stores and 6 distribution centres. "
        "Validate current performance against targets, surface key exceptions, "
        "and flag anomalies across the supply chain network."
    ),
)

MANAGER_NAME = "SupplyTowerManager"
manager = LlmAgent(
    name=MANAGER_NAME,
    llm=llm,
    tools=[AgentTool(inventory_agent), AgentTool(factpack_agent)],
    instructions=(
        "You are the Supply Chain Control Tower Manager for a large Danish grocery cooperative "
        "(member-owned, ~2 million members, ~940 stores, ~45 billion DKK annual turnover, "
        "6 distribution centres: 3 dry goods, 2 cold/fresh, 1 frozen).\n\n"
        "You MUST follow these steps in order:\n"
        "STEP 1: Call InventoryProjectionAgent to understand forward-looking inventory impact.\n"
        "STEP 2: Call LiveFactpackAgent to validate against current live performance.\n"
        "STEP 3: Synthesize a concise executive summary with clear decisions and actions.\n\n"
        "Be specific to the cooperative context. Prioritise member value and operational "
        "continuity. Do not skip steps."
    ),
    max_iterations=3,
)

runner = Runner(
    agent=manager,
    app_name="control-tower-dk",
    session_service=InMemorySessionService(),
)


# ------------------------------------------------------------
# Step definitions + renderer
# ------------------------------------------------------------
STEP_DEFS = [
    {"name": "Lagerfremskrivning",    "badge_label": "Inventory Agent", "badge_cls": "badge-inv"},
    {"name": "Live performancecheck", "badge_label": "Factpack Agent",  "badge_cls": "badge-fact"},
    {"name": "Ledelsesoversigt",      "badge_label": "Manager Agent",   "badge_cls": "badge-manager"},
]

def render_steps(placeholder):
    html = ""
    for i, step in enumerate(st.session_state.steps):
        status = step["status"]
        defn   = STEP_DEFS[i]
        bar    = '<div class="step-progress"></div>' if status == "running" else ""
        label  = {"pending": "Afventer", "running": "Kører…", "done": "Færdig"}[status]
        html += f"""
        <div class="step-card {status}">
          {bar}
          <div class="step-num {status}">{i+1}</div>
          <div class="step-info">
            <div class="step-name">{defn['name']}</div>
            <span class="agent-badge {defn['badge_cls']}">{defn['badge_label']}</span>
          </div>
          <div class="step-status status-{status}">{label}</div>
        </div>
        """
    placeholder.markdown(html, unsafe_allow_html=True)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
st.markdown("""
<div class="ct-header">
  <div>
    <h1>Forsyningskæde · Control Tower</h1>
    <div class="ct-subtitle">Multi-agent orkestrering · Realtidsanalyse · 6 distributionscentre</div>
  </div>
  <div class="live-badge">
    <div class="live-dot"></div>
    Live
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dc-strip">
  <div class="dc-pill"><span>Tørvarer</span>3 DC'er</div>
  <div class="dc-pill"><span>Kølevarer</span>2 DC'er</div>
  <div class="dc-pill"><span>Frost</span>1 DC</div>
  <div class="dc-pill"><span>Butikker</span>~940</div>
  <div class="dc-pill"><span>Kæder</span>4 formater</div>
  <div class="dc-pill"><span>Omsætning</span>~45 mia. DKK</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with right:
    st.markdown('<div class="section-label">Udførelses-pipeline</div>', unsafe_allow_html=True)
    steps_placeholder = st.empty()
    if not st.session_state.steps:
        st.session_state.steps = [{"status": "pending"} for _ in STEP_DEFS]
    render_steps(steps_placeholder)

with left:
    st.markdown('<div class="section-label">Assistent</div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Stil et spørgsmål til Control Tower…")
    if st.session_state.quick_prompt:
        user_prompt = st.session_state.quick_prompt
        st.session_state.quick_prompt = None


# ------------------------------------------------------------
# Run on prompt
# ------------------------------------------------------------
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.steps = [{"status": "pending"} for _ in STEP_DEFS]
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

            final_text = getattr(assistant_box, "_value", "") or ""
            st.session_state.messages.append(
                {"role": "assistant", "content": final_text}
            )
