# api/engine/agent_graph.py
from __future__ import annotations

import os, json, requests
from typing import TypedDict, Optional, Dict, Any, List, Tuple

from langgraph.graph import StateGraph, END

# Use your hardened explainer from agent.py (paragraph + confidence, PII-safe)
from api.engine.agent import llm_explain_structured

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

class ReviewState(TypedDict, total=False):
    application: Dict[str, Any]
    kyc: Optional[Dict[str, Any]]
    credit: Optional[Dict[str, Any]]
    aml: Optional[Dict[str, Any]]

    metrics: Dict[str, Any]                # dti, ltv, risk_score
    rule_hits: List[Dict[str, Any]]
    decision: str
    confidence: float

    # extras the UI can show
    docs_needed: List[str]
    counter_offer: Dict[str, Any]

    # LLM output
    llm_explanation: str
    explanation: str

    # compliance rollups
    kyc_status: str                        # "pass" | "fail" | "unknown"
    aml_hit: Optional[bool]                # True | False | None

    # final
    result: Dict[str, Any]

# ---------------- HTTP helpers ----------------

def _get_json(url: str, method: str = "GET", payload: Optional[dict] = None) -> Optional[dict]:
    try:
        if method.upper() == "POST":
            r = requests.post(url, json=payload, timeout=5)
        else:
            r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------------- Data derivation helpers (mirror main.py) ----------------

def _kyc_status_from_record(rec: Optional[Dict[str, Any]]) -> str:
    """
    Mock KYC JSON fields:
      - id_document_valid (bool)
      - address_match_score (0..1)
      - aml_risk_score (0..100)
      - pep_flag (bool)
    """
    if not rec:
        return "unknown"
    try:
        ok = (
            bool(rec.get("id_document_valid", True)) and
            float(rec.get("address_match_score", 1.0)) >= 0.70 and
            int(rec.get("aml_risk_score", 100)) < 70 and
            not bool(rec.get("pep_flag", False))
        )
        return "pass" if ok else "fail"
    except Exception:
        return "unknown"

def _aml_hit_from_record(rec: Optional[Dict[str, Any]]) -> Optional[bool]:
    if rec is None:
        return None
    try:
        return bool(rec.get("watchlist_hit", False))
    except Exception:
        return None

def _compute_metrics(app: Dict[str, Any]) -> Tuple[float, Optional[float]]:
    income = float(app.get("income_monthly", 0.0) or 0.0)
    debts = float(app.get("debts_monthly", 0.0) or 0.0)
    amount = float(app.get("amount", 0.0) or 0.0)
    cv_raw = app.get("collateral_value", None)
    collateral_value = float(cv_raw) if (cv_raw is not None and str(cv_raw) != "" and float(cv_raw) > 0) else None

    dti = round(debts / income, 2) if income > 0 else 0.0
    ltv = round(amount / collateral_value, 2) if collateral_value and collateral_value > 0 else None
    return dti, ltv

def _risk_score(app: Dict[str, Any], dti: float, ltv: Optional[float]) -> int:
    # identical to main.py
    score = 50
    # DTI
    if dti >= 1.0: score += 30
    elif dti >= 0.6: score += 18
    elif dti >= 0.4: score += 10
    elif dti >= 0.3: score += 5
    # LTV
    if ltv is not None:
        if ltv > 1.2: score += 25
        elif ltv > 1.0: score += 15
        elif ltv > 0.8: score += 8
    else:
        score += 5  # slight risk without collateral
    # Credit score
    cs = int(app.get("credit_score", 0) or 0)
    if cs < 550: score += 30
    elif cs < 600: score += 20
    elif cs < 650: score += 12
    elif cs < 700: score += 6
    # Employment
    emp = str(app.get("employment_status", "")).lower().strip()
    if emp in {"unemployed", "student"}:
        score += 20
    elif emp in {"contract"}:
        score += 10
    # Age
    if int(app.get("age", 0) or 0) < 21:
        score += 10
    # Amount vs income
    inc = float(app.get("income_monthly", 0) or 0)
    amt = float(app.get("amount", 0) or 0)
    if amt > inc * 20:
        score += 10
    # Purpose
    if str(app.get("purpose", "")).lower() == "personal":
        score += 5

    return max(0, min(100, score))

# ---------------- Graph nodes ----------------

def fetch_external(state: ReviewState) -> ReviewState:
    """
    Fetch mocks, but DO NOT override the form credit score (to match main.py).
    Also: NO AML fallback here (main.py has no fallback).
    """
    app = state["application"]
    cid = str(app.get("customer_id"))

    kyc = _get_json(f"{API_BASE}/mock/kyc/{cid}")
    credit = _get_json(f"{API_BASE}/mock/credit/{cid}")  # fetched for transparency only
    aml = _get_json(f"{API_BASE}/mock/aml/{cid}")        # no fallback

    ns = dict(state)
    ns["kyc"], ns["credit"], ns["aml"] = kyc, credit, aml
    return ns

def score_and_decide(state: ReviewState) -> ReviewState:
    """
    Rules MATCH main.py's decision_rules and _run_review:
      - use form credit_score
      - identical thresholds, messages, and confidence math
      - compliance overrides behave the same
    """
    app = state["application"]
    kyc_rec = state.get("kyc") or None
    aml_rec = state.get("aml") or None

    # ---- metrics
    dti, ltv = _compute_metrics(app)
    risk = _risk_score(app, dti, ltv)
    cs = int(app.get("credit_score", 600) or 600)
    income = float(app.get("income_monthly", 0.0) or 0.0)
    emp = str(app.get("employment_status", "")).strip().lower()

    reasons: List[str] = []
    hits: List[dict] = []

    def add(code: str, msg: str, sev: str):
        hits.append({"code": code, "message": msg, "severity": sev})

    # ---- HARD FAILS / FLAGS (exact text + severities like main.py)
    if cs < 500:
        msg = "Very low credit score (< 500)."
        reasons.append(msg); add("LOW_SCORE", msg, "reject")
    if income < 50_000:
        msg = "Low income (< 50,000)."
        reasons.append(msg); add("LOW_INCOME", msg, "flag")
    if emp == "unemployed":
        msg = "Unemployed status."
        reasons.append(msg); add("UNEMPLOYED", msg, "reject")
    if dti >= 0.6:
        msg = "High DTI (>= 0.6)."
        reasons.append(msg); add("HIGH_DTI", msg, "reject")
    if (ltv is not None) and (ltv > 1.2):
        msg = "High LTV (> 1.2)."
        reasons.append(msg); add("HIGH_LTV", msg, "flag")

    # If any hard reason exists, decide now (same formula)
    if reasons:
        label = "Reject" if ("High DTI (>= 0.6)." in reasons or cs < 500) else "Flag"
        confidence = max(0.05, min(0.95, 1 - risk/100))
        kyc_status = _kyc_status_from_record(kyc_rec)
        aml_hit = _aml_hit_from_record(aml_rec)

        # Compliance overrides (same as main.py)
        if kyc_status == "fail":
            label = "Reject"
            reasons = ["KYC failed (mock)."] + reasons
            confidence = min(confidence, 0.2)
            add("KYC_FAIL", "KYC failed (mock).", "reject")
        elif (aml_hit is True) and (label != "Reject"):
            label = "Flag"
            reasons = ["AML watchlist hit (mock)."] + reasons
            confidence = min(confidence, 0.5)
            add("AML_HIT", "AML watchlist hit (mock).", "flag")
        elif (kyc_status == "unknown") or (aml_hit is None):
            # only add COMPLIANCE_UNKNOWN if we flip Approve->Flag (main.py behavior)
            confidence = min(confidence, 0.6)

        ns = dict(state)
        ns["metrics"] = {"dti": dti, "ltv": ltv, "risk_score": risk}
        ns["rule_hits"] = hits
        ns["decision"] = label
        ns["confidence"] = round(float(confidence), 2)
        ns["kyc_status"] = kyc_status
        ns["aml_hit"] = aml_hit
        return ns

    # ---- POSITIVES (only if no hard reasons)
    positives: List[str] = []
    if cs >= 700:
        positives.append("Good credit score (>= 700)."); add("GOOD_SCORE", positives[-1], "info")
    if dti <= 0.4:
        positives.append("Acceptable DTI (<= 0.4)."); add("GOOD_DTI", positives[-1], "info")
    if (ltv is None) or (ltv <= 0.8):
        positives.append("Low LTV (<= 0.8) or no collateral needed."); add("GOOD_LTV", positives[-1], "info")
    if income >= 100_000:
        positives.append("High income (>= 100,000)."); add("HIGH_INCOME", positives[-1], "info")

    if (risk <= 65) and (len(positives) >= 2):
        label = "Approve"
        confidence = max(0.6, min(0.95, 1 - risk/100))
    else:
        add("BORDERLINE", "Borderline metrics; needs manual review.", "warn")
        label = "Flag"
        confidence = max(0.3, min(0.8, 1 - risk/100))

    # ---- compliance roll-up (same as main.py)
    kyc_status = _kyc_status_from_record(kyc_rec)
    aml_hit = _aml_hit_from_record(aml_rec)

    if kyc_status == "fail":
        label = "Reject"
        confidence = min(confidence, 0.2)
        reasons = ["KYC failed (mock)."] + reasons
        add("KYC_FAIL", "KYC failed (mock).", "reject")
    elif (aml_hit is True) and (label != "Reject"):
        label = "Flag"
        confidence = min(confidence, 0.5)
        reasons = ["AML watchlist hit (mock)."] + reasons
        add("AML_HIT", "AML watchlist hit (mock).", "flag")
    elif (kyc_status == "unknown") or (aml_hit is None):
        if label == "Approve":
            label = "Flag"
            reasons = ["Compliance services unavailable; safety flag."] + reasons
            add("COMPLIANCE_UNKNOWN", "Compliance services unavailable; safety flag.", "warn")
        confidence = min(confidence, 0.6)

    ns = dict(state)
    ns["metrics"] = {"dti": dti, "ltv": ltv, "risk_score": risk}
    ns["rule_hits"] = hits
    ns["decision"] = label
    ns["confidence"] = round(float(confidence), 2)
    ns["kyc_status"] = kyc_status
    ns["aml_hit"] = aml_hit
    # keep fetched credit only for transparency; decision uses form score
    ns["credit"] = state.get("credit") or {}
    return ns

def plan_documents(state: ReviewState) -> ReviewState:
    """Deterministic checklist of docs for Flag/Reject scenarios (nice extra, does not affect parity)."""
    docs: List[str] = []
    hits = state.get("rule_hits", [])
    codes = {h["code"] for h in hits}

    if "HIGH_DTI" in codes or "BORDERLINE" in codes:
        docs += ["Last 3 months salary slips or bank statements"]

    if "HIGH_LTV" in codes:
        docs += ["Collateral valuation report or proof of ownership"]

    if "LOW_SCORE" in codes:
        docs += ["Credit report consent / bureau check"]

    if "KYC_FAIL" in codes or state.get("kyc_status") == "fail":
        docs += ["Valid NIC/passport", "Recent proof of address (utility bill)"]

    if state.get("aml_hit") is True:
        docs += ["Enhanced Due Diligence (EDD) form"]

    # Dedupe while preserving order
    seen = set()
    final_docs: List[str] = []
    for d in docs:
        if d not in seen:
            final_docs.append(d)
            seen.add(d)

    ns = dict(state)
    ns["docs_needed"] = final_docs
    return ns

def suggest_counter_offer(state: ReviewState) -> ReviewState:
    """
    Suggest how to reach DTI<=0.40 and LTV<=0.80:
      - reduce_monthly_debts_by
      - additional_collateral_needed
      - or lower_loan_amount_to (if collateral can't be increased)
    """
    app = state["application"]
    dti = float(state["metrics"]["dti"])
    ltv = state["metrics"]["ltv"]
    amount = float(app.get("amount", 0.0) or 0.0)
    cv_raw = app.get("collateral_value", None)
    collateral = float(cv_raw) if (cv_raw not in (None, "",) and float(cv_raw) > 0) else 0.0
    income = float(app.get("income_monthly", 0.0) or 0.0)
    debts = float(app.get("debts_monthly", 0.0) or 0.0)

    offer: Dict[str, Any] = {}
    target_dti = 0.40
    target_ltv = 0.80

    # DTI: reduce debts to income*0.4
    if income > 0 and dti > target_dti:
        max_debts_allowed = round(income * target_dti, 2)
        reduce_by = round(max(0.0, debts - max_debts_allowed), 2)
        if reduce_by > 0:
            offer["reduce_monthly_debts_by"] = reduce_by

    # LTV: either add collateral or reduce amount
    if ltv is not None and ltv > target_ltv:
        additional_collateral = round((amount / target_ltv) - collateral, 2)
        if additional_collateral > 0:
            offer["additional_collateral_needed"] = additional_collateral

        max_amount_at_target = round(collateral * target_ltv, 2)
        if amount > max_amount_at_target:
            offer["lower_loan_amount_to"] = max_amount_at_target

    ns = dict(state)
    ns["counter_offer"] = offer
    return ns

def explain_with_llm(state: ReviewState) -> ReviewState:
    """
    Use the same context shape the UI expects, and read the form's credit_score
    (to match main.py). LLM text only; no decision change here.
    """
    app = state.get("application", {}) or {}
    review_result = {
        # decision + scores
        "decision": state.get("decision"),
        "dti": state.get("metrics", {}).get("dti"),
        "ltv": state.get("metrics", {}).get("ltv"),
        "credit_score": app.get("credit_score"),  # <-- form value (parity with main.py)
        "kyc_status": state.get("kyc_status"),
        "aml_hit": state.get("aml_hit"),
        "risk_score": state.get("metrics", {}).get("risk_score"),
        "reasons": [h.get("message") for h in state.get("rule_hits", []) if h.get("message")],
        "rule_hits": state.get("rule_hits", []),

        # extras for a nicer paragraph in your UI
        "docs_needed": state.get("docs_needed", []),
        "counter_offer": state.get("counter_offer", {}),

        # application context (shown in UI)
        "age": app.get("age"),
        "employment_status": app.get("employment_status"),
        "income_monthly": app.get("income_monthly"),
        "debts_monthly": app.get("debts_monthly"),
        "amount": app.get("amount"),
        "term_months": app.get("term_months"),
        "collateral_value": app.get("collateral_value"),
        "purpose": app.get("purpose"),
    }

    out = llm_explain_structured(review_result)
    explanation_text = out.get("explanation", "") or ""

    ns = dict(state)
    ns["llm_explanation"] = explanation_text
    ns["explanation"] = explanation_text  # back-compat

    # blend LLM confidence if higher
    try:
        llm_conf = float(out.get("confidence", 0) or 0.0)
    except Exception:
        llm_conf = 0.0
    try:
        current_conf = float(ns.get("confidence", 0) or 0.0)
    except Exception:
        current_conf = 0.0
    ns["confidence"] = max(current_conf, llm_conf)

    return ns

def finalize(state: ReviewState) -> ReviewState:
    res = {
        "decision": state["decision"],
        "confidence": state["confidence"],
        "dti": state["metrics"]["dti"],
        "ltv": state["metrics"]["ltv"],
        "risk_score": state["metrics"]["risk_score"],
        "reasons": [h["message"] for h in state.get("rule_hits", []) if h.get("severity") in ("warn", "flag", "reject")],
        "rule_hits": state.get("rule_hits", []),

        # expose both keys; UI prefers llm_explanation but falls back to explanation
        "llm_explanation": state.get("llm_explanation", ""),
        "explanation": state.get("llm_explanation", "") or state.get("explanation", ""),

        "kyc_status": state.get("kyc_status", "unknown"),
        "aml_hit": state.get("aml_hit", None),

        # extra agent goodies (optional for UI)
        "docs_needed": state.get("docs_needed", []),
        "counter_offer": state.get("counter_offer", {}),
    }
    ns = dict(state)
    ns["result"] = res
    return ns

def build_graph():
    g = StateGraph(ReviewState)

    g.add_node("fetch_external", fetch_external)
    g.add_node("score_and_decide", score_and_decide)
    g.add_node("plan_documents", plan_documents)
    g.add_node("suggest_counter_offer", suggest_counter_offer)
    g.add_node("explain_with_llm", explain_with_llm)
    g.add_node("finalize", finalize)

    g.set_entry_point("fetch_external")
    g.add_edge("fetch_external", "score_and_decide")
    g.add_edge("score_and_decide", "plan_documents")
    g.add_edge("plan_documents", "suggest_counter_offer")
    g.add_edge("suggest_counter_offer", "explain_with_llm")
    g.add_edge("explain_with_llm", "finalize")
    g.add_edge("finalize", END)

    return g.compile()

GRAPH = build_graph()

def run_review_with_graph(application_payload: Dict[str, Any]) -> Dict[str, Any]:
    state: ReviewState = {"application": application_payload}
    out = GRAPH.invoke(state)
    return out["result"]
