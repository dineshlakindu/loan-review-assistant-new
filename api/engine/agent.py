# api/engine/agent.py
# Natural-language LLM explainer for loan decisions via Ollama

from __future__ import annotations

import os
import re
import json
import requests
from typing import Dict, Any, List, Tuple

# ---- Ollama config (override via environment) ----
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# Examples: "llama3.2:3b", "mistral", fall back to tiny if needed
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Keep outputs steady and discourage lists
GENERATION_OPTIONS: Dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    # a bit more budget so the model can write a fuller paragraph
    "num_predict": 360,
    # Block common bullet/numbered list starts
    "stop": ["\n- ", "\n* ", "\n• ", "\n1. ", "\n1) ", "\n2. ", "\n2) "],
}

# ---------- Internal helpers ----------

def _check_ollama_alive() -> None:
    """Raise if Ollama is not reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}. Details: {e}")

def _format_fallback(review_result: Dict[str, Any]) -> str:
    """Deterministic paragraph if LLM is unavailable."""
    d = review_result or {}
    reasons: List[str] = d.get("reasons") or []
    score = d.get("score") or d.get("credit_score") or "N/A"
    kyc = d.get("kyc_status") or d.get("kyc") or "N/A"
    dti = d.get("dti", "N/A")
    ltv = d.get("ltv", "N/A")
    expl = (
        f"The decision is {d.get('decision','N/A')}. "
        f"The credit score is {score}, the debt-to-income ratio is {dti}, and the loan-to-value is {ltv}. "
        f"KYC/AML status is {kyc}. "
        f"Key reasons include: {', '.join(reasons) if reasons else 'not specified'}. "
        f"This summary is shown because the local LLM is unavailable."
    )
    next_step = _infer_next_step(d)
    return f"{expl} Next step: {next_step}"

def _postprocess_to_one_paragraph(text: str) -> str:
    text = re.sub(r'(?mi)^\s*(decision|result|summary)\s*:\s*', '', text)
    text = re.sub(r'(?m)^\s*[-*•]\s*', '', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()
    text = ' '.join(text.split())
    return text

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _derive_confidence(review_result: Dict[str, Any]) -> float:
    """
    If caller didn't compute confidence, produce a reasonable heuristic:
    - prefer provided 'confidence' (0..1 or 0..100)
    - otherwise, use risk_score and rule severities if available
    """
    # prefer provided
    if "confidence" in review_result and review_result["confidence"] is not None:
        c = float(review_result["confidence"])
        # accept 0..1 or 0..100
        return _clamp(c if c <= 1.0 else c / 100.0)

    # heuristic from risk_score and rule severities
    risk = float(review_result.get("risk_score", 60) or 60.0)  # typical mid
    base = 0.5 + (risk - 50.0) / 100.0  # ~0.4..0.6 around 50
    rule_hits = review_result.get("rule_hits") or []
    # boost for infos, penalize warns/errors
    for rh in rule_hits:
        sev = str(rh.get("severity", "")).lower()
        if sev == "info":
            base += 0.03
        elif sev == "warn":
            base -= 0.05
        elif sev == "error":
            base -= 0.12
    return _clamp(base, 0.25, 0.95)

# --- robust normalization for decision/kyc tokens
def _norm_token(s: Any) -> str:
    return re.sub(r'[^a-z]', '', str(s).lower())

def _infer_next_step(d: Dict[str, Any]) -> str:
    """Heuristic 'Next step' so UI always shows an action, even if model omits it."""
    decision = _norm_token(d.get("decision", ""))
    kyc = _norm_token(d.get("kyc_status") or d.get("kyc") or "")
    score = d.get("score") or d.get("credit_score")
    dti = d.get("dti")
    ltv = d.get("ltv")
    reasons = [str(r).lower() for r in (d.get("reasons") or [])]

    # Look into rule hits too
    rule_hits = d.get("rule_hits") or []
    hit_codes = {_norm_token(h.get("code", "")) for h in rule_hits}
    hit_sev = {_norm_token(h.get("severity", "")) for h in rule_hits}

    # KYC problems dominate next step
    if kyc in {"fail", "failed", "reject"} or "kycfail" in hit_codes:
        return "Contact the applicant to re-submit valid KYC documents and re-verify; escalate to Compliance if mismatches persist."

    # High DTI / affordability
    if isinstance(dti, (int, float)) and dti is not None and dti >= 0.40:
        return "Request updated income proofs or propose a smaller amount/longer term to bring DTI under 0.40."

    # High LTV / collateral short
    if isinstance(ltv, (int, float)) and ltv is not None and ltv > 0.80:
        return "Ask for a higher down payment or additional collateral to reduce LTV below 0.80."

    # Low score
    if isinstance(score, (int, float)) and score is not None and score < 620:
        return "Offer a credit-builder path or consider a guarantor/secured product; re-evaluate after improvements."

    # Borderline/warn
    if "warn" in hit_sev or "borderline" in reasons or "borderline" in hit_codes:
        return "Send for manual analyst review with focus on borderline metrics and recent income trends."

    # Decision-specific fallbacks (robust to synonyms and casing)
    if decision in {"approve", "approved"}:
        return "Issue the approval letter, complete the disbursement checklist, and schedule repayment auto-debit."
    if decision in {"reject", "rejected", "decline", "declined"}:
        return "Send a decline letter with key reasons and suggest remediation (improve score, reduce amount, add collateral)."
    if decision in {"flag", "flagged", "review"}:
        return "Queue for manual review and verify the documents noted in the rule trace."

    # Generic fallback
    return "Communicate the decision with reasons and suggest options (reduce amount, add collateral, or re-verify documents)."

def _ensure_next_step(text: str, d: Dict[str, Any]) -> str:
    """Append 'Next step:' if the model forgot it."""
    if re.search(r'(?i)\bnext step\s*:', text):
        return text
    return f"{text} Next step: {_infer_next_step(d)}"

# ---------- Public API (v1: string only, kept for backward-compat) ----------

def llm_explain(review_result: Dict[str, Any]) -> str:
    """
    Explain the loan decision in one cohesive paragraph (5–7 sentences) of natural ENGLISH,
    then add a final 'Next step:' action.
    No headings, no bullet points, no numbered lists, no JSON.
    """
    prompt = (
        "Write ONE cohesive paragraph (5–7 sentences) in ENGLISH ONLY that explains this loan decision "
        "(Approved, Rejected, or Flagged) in a natural, human tone. "
        "Briefly interpret what the key metrics imply (e.g., what a given DTI or LTV means versus typical guidelines) "
        "without inventing any numbers. Weave in DTI, LTV, credit score, risk or rule hits, and KYC/AML status only if present. "
        "Avoid lists, tables, markdown, or headings. After the paragraph, add ONE final line that starts exactly with "
        "'Next step:' followed by a concise recommended action for the reviewer.\n\n"
        f"Decision JSON:\n{json.dumps(review_result, ensure_ascii=False, indent=2)}"
    )

    try:
        _check_ollama_alive()
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional loan officer. "
                            "Produce clear, natural ENGLISH explanations as a single cohesive paragraph (5–7 sentences). "
                            "Never output bullet points, lists, JSON, tables, or headings. "
                            "Always end with a final line that begins with 'Next step:'."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": GENERATION_OPTIONS,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "").strip()
        text = _postprocess_to_one_paragraph(text)
        return _ensure_next_step(text, review_result) or "LLM returned empty text."
    except Exception:
        return _format_fallback(review_result)

# ---------- Public API (v2: structured explanation + confidence) ----------

def llm_explain_structured(review_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: {"explanation": str, "confidence": float (0..1)}
    - Requests JSON from the model; falls back gracefully.
    - If model returns plain text, we still fill both fields.
    - We append a 'Next step:' line to the explanation so the UI always shows it.
    """
    system = (
        "You are a concise, professional loan officer. "
        "Explain decisions in one cohesive paragraph (5–7 sentences). "
        "No lists, no headings. Output JSON only with keys: explanation (string, paragraph), next_step (string), confidence (0..1)."
    )
    user = {
        "instruction": (
            "Explain the decision briefly in natural ENGLISH and output pure JSON with keys "
            "`explanation` (string, one cohesive paragraph), `next_step` (string), and `confidence` (0..1)."
        ),
        "review_result": review_result,
        "constraints": [
            "Use only values given in review_result (no invented numbers).",
            "Avoid lists, markdown, or headings.",
            "The explanation should be 5–7 sentences in a natural tone."
        ],
        "schema": {"explanation": "string", "next_step": "string", "confidence": "number (0..1)"}
    }

    # default fallback
    fallback_text_only = _format_fallback(review_result)  # already includes Next step
    fallback_conf = _derive_confidence(review_result)

    try:
        _check_ollama_alive()
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                "stream": False,
                "options": GENERATION_OPTIONS,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "").strip()

        # try JSON first
        try:
            data = json.loads(content)
            expl = _postprocess_to_one_paragraph(str(data.get("explanation", "")).strip())
            model_next = str(data.get("next_step", "")).strip()
            conf = data.get("confidence", None)
            if conf is not None:
                conf = float(conf)
                conf = conf if conf <= 1.0 else conf / 100.0
            else:
                conf = fallback_conf

            if not expl:
                # if explanation is empty, try postprocessing plain text
                expl = _postprocess_to_one_paragraph(content)

            if not model_next:
                model_next = _infer_next_step(review_result)

            final_expl = _ensure_next_step(expl, {**review_result, "next_step": model_next})
            # Ensure the explicit next_step appears even if the ensure fn doesn't see it
            if "next step:" not in final_expl.lower():
                final_expl = f"{final_expl} Next step: {model_next}"

            return {"explanation": final_expl or fallback_text_only, "confidence": _clamp(conf)}
        except Exception:
            # model returned plain text
            text = _postprocess_to_one_paragraph(content) or fallback_text_only
            text = _ensure_next_step(text, review_result)
            return {"explanation": text, "confidence": fallback_conf}
    except Exception:
        # Ollama down
        return {"explanation": fallback_text_only, "confidence": fallback_conf}
