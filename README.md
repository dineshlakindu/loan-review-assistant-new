![CI](https://github.com/dineshlakindu/loan-review-assistant/actions/workflows/ci.yml/badge.svg?branch=main)

# Loan Review Assistant (IS4007 Demo)

A small single-agent system that reviews **synthetic loan applications** using **rules + local LLM explanation**, with a **FastAPI** backend and **Streamlit** UI.  
Simulated external systems (mock APIs): **KYC / Credit / AML**. A LangGraph agent route is included and safely disabled in CI if LangGraph isn’t installed.

---

## Features
- **FastAPI** service with typed schemas and a JSON-safe `/decisions` log
- **Streamlit** UI: form entry, CSV autofill, explanations, recent decisions table
- **Mock services**: `/mock/kyc/{id}`, `/mock/credit/{id}`, `/mock/aml/{id}`
- **Agent route** (optional): `/graph/review` via LangGraph
- **Tests + CI**: `pytest` with GitHub Actions badge above

---

![CI](https://github.com/dineshlakindu/loan-review-assistant/actions/workflows/ci.yml/badge.svg?branch=main)

# Loan Review Assistant (IS4007 Demo)

A small single-agent system that reviews **synthetic loan applications** using **rules + local LLM explanation**, with a **FastAPI** backend and **Streamlit** UI.  
Simulated external systems (mock APIs): **KYC / Credit / AML**. A LangGraph agent route is included and safely disabled in CI if LangGraph isn’t installed.

---

## Features
- **FastAPI** service with typed schemas and a JSON-safe `/decisions` log
- **Streamlit** UI: form entry, CSV autofill, explanations, recent decisions table
- **Mock services**: `/mock/kyc/{id}`, `/mock/credit/{id}`, `/mock/aml/{id}`
- **Agent route** (optional): `/graph/review` via LangGraph
- **Tests + CI**: `pytest` with GitHub Actions badge above

---

# README — Setup & Run on a New Laptop (Windows / macOS / Linux)

This guide shows only how to **install**, **run**, and **verify** the Loan Review Assistant on a fresh machine.

---

## 1) Prerequisites

- **Python 3.10+**
- **Git** (only needed if you plan to clone the repo)
- **Ollama** (local LLM runtime) — install from https://ollama.com

After installing Ollama, open a terminal and pull a model:
```bash
ollama pull llama3.2:3b
# (or: ollama pull mistral)
2) Get the code (choose ONE option)
Option A — Clone from GitHub
bash
Copy code
git clone https://github.com/tharusharanaweera/loan_review_assistant.git
cd loan_review_assistant
Option B — You ALREADY have all files locally
If the project folder (with api/, ui/, data/, etc.) already exists on your laptop (e.g., copied from USB/ZIP), open a terminal in that folder and start from Step 3 below.

Tip: On Windows Explorer, Shift + right-click the folder → “Open PowerShell window here”. On macOS Finder, right-click → “New Terminal at Folder”.

3) Create & activate a virtual environment
Which shell am I using?

PowerShell usually shows PS C:\...>

Git Bash shows MINGW64 ... $

macOS/Linux Terminal shows $ without MINGW64

Windows (PowerShell)
powershell
Copy code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked, run this once in the SAME PowerShell:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Windows (Git Bash)
bash
Copy code
python -m venv .venv
# IMPORTANT: On Windows, Git Bash uses Scripts/, not bin/
source .venv/Scripts/activate
macOS / Linux (Terminal)
bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
4) Install dependencies
bash
Copy code
pip install -r requirements.txt
5) Configure environment
Copy the example env file and edit if needed:

bash
Copy code
# from the repo root
cp .env.example .env
.env (defaults are fine for local run):

ini
Copy code
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
API_HOST=127.0.0.1
API_PORT=8000
6) Start Ollama (LLM server)
Open a separate terminal and run:

bash
Copy code
ollama serve
Keep this running. (The API will call it at http://127.0.0.1:11434.)

7) Run the API (FastAPI)
Back in your virtualenv terminal:

bash
Copy code
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
Check it: open http://127.0.0.1:8000/docs in your browser.

8) Run the UI (Streamlit)
Open a new terminal, activate the same venv (see Step 3), then:

bash
Copy code
streamlit run ui/app.py
Open the link printed in the terminal (usually http://localhost:8501).
Fill the form and click Review to see a decision + explanation.

9) Quick verification (optional)
With the API running, send a sample request.

Windows PowerShell (one line):

powershell
Copy code
curl -Method POST http://127.0.0.1:8000/graph/review -ContentType "application/json" -Body "{""applicant_id"":""CUST-001"",""amount"":1200000,""term_months"":36,""income_monthly"":150000,""debts_monthly"":60000,""employment_status"":""employed"",""credit_score"":720,""collateral_value"":1500000}"
macOS/Linux/Git Bash:

bash
Copy code
curl -X POST "http://127.0.0.1:8000/graph/review" \
  -H "Content-Type: application/json" \
  -d '{ "applicant_id":"CUST-001", "amount":1200000, "term_months":36, "income_monthly":150000, "debts_monthly":60000, "employment_status":"employed", "credit_score":720, "collateral_value":1500000 }'
You should get JSON with decision, metrics, confidence, and explanation.

10) Where results are logged
Decisions are appended to:

bash
Copy code
data/decisions_log.csv
Troubleshooting
Ollama connection error

Ensure ollama serve is running.

Confirm OLLAMA_URL in .env.

Pull a model: ollama pull llama3.2:3b (or mistral).

Windows venv activation blocked (PowerShell only)

In the same PowerShell session:

powershell
Copy code
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Then run .\.venv\Scripts\Activate.ps1 again.

Wrong activate path in Git Bash

Use source .venv/Scripts/activate (Windows Git Bash)
not source .venv/bin/activate (that’s for macOS/Linux).

Port already in use

API: uvicorn api.main:app --port 8001 --reload

UI: streamlit run ui/app.py --server.port 8502

CSV/files not found

Run commands from the repo root so relative paths resolve.

Ensure data/kyc.csv, data/aml.csv, data/customers.csv exist.

CORS (UI cannot call API)

Ensure the API is running at 127.0.0.1:8000.

If you changed ports, update the UI config (if applicable).

Stop services
Press Ctrl+C in each terminal to stop UI, API, and Ollama.

Deactivate venv:

macOS/Linux/Git Bash: deactivate

PowerShell: deactivate

Done! Whether you cloned from GitHub or already had all files locally, the steps above will get the API and UI running.

Copy code
