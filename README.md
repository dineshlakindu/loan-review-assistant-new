# Loan Review Assistant
FastAPI + Streamlit + Ollama demo that reviews synthetic loan applications,
applies rules (DTI, LTV, score, KYC), and produces a natural-language explanation
with a final **Next step** action.
1) Start API: `uvicorn api.main:app --reload`
2) Start UI:  `streamlit run ui/app.py`
