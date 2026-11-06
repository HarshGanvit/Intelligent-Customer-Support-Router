"""Routing rules shared by the training notebook and app.py (single source of truth)."""
import re

GENERAL = "GENERAL"

DEFAULT_CONFIG = {
    "escalate_crit_prob": 0.50,   # P(critical) >= this -> Tier 1
    "review_crit_prob": 0.25,     # P(critical) in [review, escalate) -> Tier 2 priority ticket
    "routine_min_prob": 0.50,     # P(routine) needed before the bot may answer / ticket as routine
    "frustrated_prob": 0.60,      # P(frustrated) >= this -> Tier 2
    "greeting_min_prob": 0.50,
    "dept_min_conf": 0.40,        # below this the retrieval search ignores the predicted department
    "retrieval_min_sim": 0.55,    # overwritten by the calibrated value
}

# ------------------------------------------------------------------ placeholder cleaning (Issue 6)
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_NUMBERED = {"order number": "order", "invoice number": "invoice", "tracking number": "shipment"}
_DEFAULTS = {
    "website url": "our website", "customer support phone number": "our support line",
    "customer support email": "our support email", "customer support hours": "our support hours",
    "online company portal info": "our online portal", "online order interaction": "your online account",
    "store address": "our nearest store", "account category": "your account", "account type": "your account",
}
_KEYWORD_DEFAULTS = [("url", "our website"), ("website", "our website"), ("phone", "our support line"),
                     ("email", "our support email"), ("hours", "our support hours"), ("address", "our nearest store"),
                     ("location", "our nearest store"), ("name", "valued customer"), ("date", "the expected date"),
                     ("time", "the scheduled time"), ("amount", "the amount"), ("portal", "our online portal"),
                     ("account", "your account"), ("settings", "your account settings")]
_ORDER_IN_TEXT = re.compile(r"(?:order|#)\s*(?:number|no\.?)?\s*[:#]?\s*((?=[A-Z0-9-]*\d)[A-Z0-9-]{4,})", re.I)


def extract_slots(user_text):
    """Pull slot values (currently the order number) out of what the customer wrote."""
    m = _ORDER_IN_TEXT.search(user_text or "")
    return {"order number": m.group(1)} if m else {}


def clean_response(text, context=None):
    """Replace every {{Placeholder}} so customers never see raw template tags."""
    text = str(text)
    ctx = {k.lower(): str(v) for k, v in (context or {}).items() if v}

    for key, noun in _NUMBERED.items():
        pat = re.compile(r"(?:\b(?:your|the)\s+)?(?:\b%s\s*(?:number|no\.?|#)?\s*)?\{\{\s*%s\s*\}\}" % (noun, re.escape(key)), re.I)
        base = f"your {noun} {ctx[key]}" if key in ctx else f"your {noun}"

        def _sub(m, base=base):
            start = m.start()
            sentence_start = start == 0 or m.string[max(0, start - 2):start] in (". ", "! ", "? ")
            return base[0].upper() + base[1:] if sentence_start else base
        text = pat.sub(_sub, text)

    def _generic(m):
        key = m.group(1).strip().lower()
        if key in ctx:
            return ctx[key]
        if key in _DEFAULTS:
            return _DEFAULTS[key]
        for kw, val in _KEYWORD_DEFAULTS:
            if kw in key:
                return val
        return key

    text = _PLACEHOLDER.sub(_generic, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# ------------------------------------------------------------------ retrieval gate (Issue 3)
def best_match(query_emb, pool_embs, pool_categories, dept, dept_conf, cfg=None):
    """Index + cosine similarity of the closest pool item. Restricts to the predicted department when we trust it."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    import numpy as np
    mask = np.ones(len(pool_categories), dtype=bool)
    if dept != GENERAL and dept_conf >= cfg["dept_min_conf"]:
        m = np.asarray(pool_categories) == dept
        if m.any():
            mask = m
    idx = np.where(mask)[0]
    sims = pool_embs[idx] @ query_emb
    j = int(sims.argmax())
    return int(idx[j]), float(sims[j])


# ------------------------------------------------------------------ the router (Issue 2)
def route_ticket(pred, retrieve_fn=None, create_ticket_fn=None, cfg=None):
    """pred needs: text, department, department_confidence, urgency_probs{}, sentiment_probs{}.
    retrieve_fn(text, dept, dept_conf) -> {"reply", "similarity", "matched_instruction"} or None
    create_ticket_fn(text, dept, urgency, sentiment, priority) -> ticket id
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    text, dept = pred["text"], pred["department"]
    dept_conf = pred.get("department_confidence", 1.0)
    pu, ps = pred["urgency_probs"], pred["sentiment_probs"]
    p_crit, p_frus, p_rout = pu.get("critical", 0.0), ps.get("frustrated", 0.0), pu.get("routine", 0.0)
    top_urg = max(pu, key=pu.get)
    sent_label = "frustrated" if p_frus >= cfg["frustrated_prob"] else max(ps, key=ps.get)

    def out(tier, action, reply, reason, ticket_id=None, similarity=None, matched=None):
        return {"tier": tier, "action": action, "reply": reply, "reason": reason, "ticket_id": ticket_id,
                "similarity": similarity, "matched_instruction": matched}

    def ticket(urgency, priority):
        return create_ticket_fn(text, dept, urgency, sent_label, priority) if create_ticket_fn else "TCK-DRYRUN"

    # Tier 1 - likely critical, regardless of tone
    if p_crit >= cfg["escalate_crit_prob"]:
        who = "a human agent" if dept == GENERAL else f"a human agent ({dept})"
        return out(1, "escalate_to_human", f"I understand this is urgent. I'm connecting you to {who} right now.", "critical")

    # Possibly critical but not sure -> never let it fall through to "I didn't catch that"
    if p_crit >= cfg["review_crit_prob"] and top_urg in ("critical", "routine"):
        tid = ticket("critical?", "high")
        return out(2, "empathy_and_ticket",
                   f"I want to make sure this gets the right attention. I've created priority ticket {tid} and a team member will review it shortly.",
                   "possible_critical", tid)

    if top_urg == "greeting" and pu.get("greeting", 0.0) >= cfg["greeting_min_prob"]:
        return out(4, "greet", "Hi! How can I help you today?", "greeting")

    if top_urg == "nonsense" or dept == GENERAL:
        return out(4, "clarify", "I can help with orders, payments, refunds, shipping, invoices and accounts. What do you need help with?", "out_of_scope")

    if p_rout >= cfg["routine_min_prob"]:
        # Tier 2 - routine but frustrated
        if p_frus >= cfg["frustrated_prob"]:
            tid = ticket("routine", "normal")
            return out(2, "empathy_and_ticket",
                       f"I'm really sorry about the frustration - that's not the experience we want you to have. "
                       f"I've created ticket {tid} and a team member will review it and get back to you.", "routine_frustrated", tid)
        # Tier 3 - only with a real match
        if retrieve_fn is None:
            return out(3, "bot_resolves", "(bot answer goes here)", "routine_calm")
        hit = retrieve_fn(text, dept, dept_conf)
        if hit and hit["similarity"] >= cfg["retrieval_min_sim"]:
            ctx = extract_slots(text)
            return out(3, "bot_resolves", clean_response(hit["reply"], ctx), "routine_calm", None, hit["similarity"], hit.get("matched_instruction"))
        sim = hit["similarity"] if hit else None
        return out(4, "clarify", "I'm not sure I have the right answer for that. Could you rephrase it, or tell me a bit more about what you need?",
                   "low_retrieval_similarity", None, sim)

    return out(4, "clarify", "I didn't quite catch that. Could you describe your issue?", "low_confidence")
