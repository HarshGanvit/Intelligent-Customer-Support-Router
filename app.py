"""Intelligent Customer Support Router - Streamlit app (simplified UI).   Run:  streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

from routing import route_ticket

st.set_page_config(page_title="Customer Support Router", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(HERE, "router_artifacts")      # fixed location (the settings sidebar is hidden)

TIERS = {
    1: ("Tier 1 - Escalate to human", "error"),
    2: ("Tier 2 - Empathy + ticket", "warning"),
    3: ("Tier 3 - Bot resolves", "success"),
    4: ("Tier 4 - Greet / clarify", "info"),
}
EXAMPLES = [
    "What payment methods do you accept for orders?",
    "Hello! how can i get refund",
    "How do I recover my password?",
    "how do I change my shipping address",
    "WHY WAS MY CARD CHARGED THREE TIMES FOR ONE ORDER?! REFUND MY MONEY RIGHT NOW!",
    "hello",
    "Can you tell me a joke?",
    "asdfgh qwerty",
]

@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_engine(art_dir):
    from inference import RouterEngine          # imported lazily so a missing torch gives a clear message
    return RouterEngine(art_dir)


# ------------------------------------------------------------------ session state
st.session_state.setdefault("history", [])           # oldest -> newest
st.session_state.setdefault("tickets", [])
st.session_state.setdefault("show_suggestions", False)
st.session_state.setdefault("pending", None)


def create_ticket(text, department, urgency, sentiment, priority="normal"):
    tid = f"TCK-{len(st.session_state.tickets) + 1:05d}"
    st.session_state.tickets.append({"id": tid, "priority": priority, "department": department,
                                     "urgency": urgency, "sentiment": sentiment, "message": text})
    return tid


def probs_to_sentence(probs):
    """{'frustrated': 0.22, 'neutral': 0.78} -> 'neutral 78%, frustrated 22%' (highest first)."""
    items = probs.items() if isinstance(probs, dict) else enumerate(probs)
    items = sorted(items, key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{k} {v * 100:.0f}%" for k, v in items)


# ------------------------------------------------------------------ model (no sidebar, thresholds stay at the model's defaults)
if not os.path.exists(os.path.join(ART_DIR, "router_weights.pt")):
    st.error(f"No trained model found in `{ART_DIR}`. Run the notebook to the end and unzip its output next to app.py.")
    st.stop()

try:
    engine = load_engine(ART_DIR)
except Exception as exc:                          # noqa: BLE001 - show the real reason in the UI
    st.error(f"Could not load the model: {exc}")
    st.stop()

cfg = dict(engine.cfg)

# ------------------------------------------------------------------ main (chat only)
st.title("Customer Support Router")

# --- suggestions on/off button
label = "Hide suggestions" if st.session_state.show_suggestions else "💡 Suggest messages"
if st.button(label):
    st.session_state.show_suggestions = not st.session_state.show_suggestions
    st.rerun()

if st.session_state.show_suggestions:
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}"):
            st.session_state.pending = ex              # clicking a suggestion sends it straight away

# --- chat input (typed message OR clicked suggestion)
typed = st.chat_input("Type a customer message...")
text = typed or st.session_state.pending
st.session_state.pending = None

if text and text.strip():
    pred = engine.predict(text.strip())
    result = route_ticket(pred, retrieve_fn=lambda t, d, c: engine.retrieve(t, d, c, cfg),
                          create_ticket_fn=create_ticket, cfg=cfg)
    st.session_state.history.append({**pred, **result})

# --- conversation (oldest first, last 20)
for item in st.session_state.history[-20:]:
    with st.chat_message("user"):
        st.write(item["text"])
    with st.chat_message("assistant"):
        st.write(item["reply"])
        tier_label, kind = TIERS[item["tier"]]
        getattr(st, kind)(f"{tier_label}  ·  action: {item['action']}  ·  reason: {item['reason']}")
        st.caption(
            f"Dept = {item['department']} ({item['department_confidence'] * 100:.0f}%)  |  "
            f"Urgency: {probs_to_sentence(item['urgency_probs'])}  |  "
            f"Sentiment: {probs_to_sentence(item['sentiment_probs'])}"
        )
        if item.get("ticket_id"):
            st.caption(f"🎫 Ticket created: {item['ticket_id']}")

# ------------------------------------------------------------------ sidebar: tickets
# (written after the routing code above so a ticket created by this message shows up immediately)
with st.sidebar:
    st.title("🎫 Tickets")
    tickets = st.session_state.tickets
    if not tickets:
        st.caption("No tickets yet. Tier 2 messages create one.")
    else:
        st.caption(f"{len(tickets)} open")
        for t in reversed(tickets):                    # newest first
            with st.expander(f"{t['id']} · {t['priority']} · {t['department']}"):
                st.write(f"**Urgency:** {t['urgency']}")
                st.write(f"**Sentiment:** {t['sentiment']}")
                st.write(f"**Message:** {t['message']}")
        if st.button("Clear tickets"):
            st.session_state.tickets = []
            st.rerun()
