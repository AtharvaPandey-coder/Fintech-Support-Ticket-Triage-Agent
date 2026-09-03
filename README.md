# Fintech Support Ticket Triage Agent

An agentic system that reads a fintech customer complaint, classifies it, retrieves the relevant company policy, drafts a grounded reply, checks that reply for hallucination, and decides whether it's safe to auto-resolve or should be escalated to a human — with a clear, written reason either way.

Built as a project submission for the Razorpay AI Buildathon (2026).

## Problem

Fintech support teams handle thousands of complaints daily — failed payments, KYC issues, debt collection disputes, credit reporting errors. Manually triaging every ticket is slow and inconsistent, but blindly automating all of it is risky: some categories (chargebacks, debt collection, fraud) legally require human review. This project automates what's safe to automate, and is explicit about what isn't.

## Architecture

1. **Classification (3-way comparison)** — a real dataset (CFPB Consumer Complaint Database, ~114K labeled complaints) is used to compare:
   - Keyword baseline: 42% accuracy
   - Trained ML (TF-IDF + Logistic Regression): **85% accuracy**
   - LLM zero-shot classification: 76% accuracy
   
   The trained model outperforms both simpler and more expensive alternatives — a deliberate, evidence-based choice, not just "use an LLM for everything."

2. **Sentiment/Urgency (VADER)** — tags ticket tone (angry/frustrated/neutral) to inform urgency alongside the predicted category.

3. **Policy Knowledge Base** — 43 structured policy documents across 11 fintech categories (mortgage, credit card, debt collection, credit reporting, etc.), grounded in real regulatory concepts (Regulation E, ECOA, FCRA).

4. **RAG Retrieval** — sentence-transformer embeddings (`all-MiniLM-L6-v2`) + FAISS, running fully locally with no external API dependency for retrieval.

5. **Agent — Draft + Self-Critique** — an LLM (via Groq) drafts a reply grounded only in the retrieved policy. A second LLM call then checks that draft against the same policy for contradictions or invented details before it's ever shown to anyone.

6. **Decision Logic** — combines classifier confidence, category risk level, and the self-critique result to decide: auto-resolve, or escalate with a specific written reason. High-risk categories (e.g., debt collection) always escalate, regardless of confidence.

7. **Streamlit UI** — paste a ticket, see the full decision trail: category, confidence, matched policy, self-critique verdict, and final action.

## Tech Stack

- **ML/NLP:** scikit-learn (TF-IDF, Logistic Regression, SVM), VADER
- **LLM:** Groq API (`openai/gpt-oss-20b`)
- **RAG:** LangChain, sentence-transformers, FAISS
- **UI:** Streamlit

## Key Design Decisions

- **Trained ML over LLM-only classification** — proven with a real accuracy comparison, not assumed.
- **Local embeddings, not a hosted API** — for reliability; no network dependency during the actual demo.
- **Self-critique before auto-send** — the differentiator most naive LLM-wrapper submissions skip.
- **Hard-coded escalation for regulated categories** — confidence alone never overrides compliance-sensitive cases.
