# LinkedIn Assistant — Explainable, Open-Weight LLM Job Matching

CSE 573 (Fall 2026), Group 3. See `IMPLEMENTATION_STEPS.md` for the roadmap and `DECISIONS.md` for the decision log.

## Setup (macOS arm64; Linux is similar)

Prerequisites: Homebrew, conda, Docker Desktop.

```bash
# 1. Python environment
conda create -n linkedin-assistant python=3.11 -y
conda activate linkedin-assistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Secrets
cp .env.example .env        # then set NEO4J_PASSWORD, GROQ_API_KEY, OPENROUTER_API_KEY

# 3. Neo4j 5.26 + APOC + GDS (Louvain), in Docker
docker compose --env-file .env up -d     # browser at http://localhost:7474

# 4. Local LLMs
brew install ollama && brew services start ollama
ollama pull llama3.1:8b && ollama pull qwen2.5:7b && ollama pull gemma2:9b

# 5. Kaggle credentials (for dataset download)
kaggle auth login

# 6. Verify everything
python scripts/check_env.py
```

QLoRA training (R5) needs CUDA, so run it on Colab/Kaggle with `pip install -r requirements-gpu.txt`.

## Layout

| Folder | Thread |
|---|---|
| `data/` | raw / interim / processed / external (git-ignored) |
| `ontology/` | R1: JSON Schema + Pydantic models |
| `extraction/` | R2, R5: CRF baseline, LLM extraction, LoRA |
| `graph/` | R3: normalization, Neo4j load |
| `ranking/` | R4: lexical, dense, RRF, LambdaMART, cross-encoder |
| `router/` | R7: model router + call logs |
| `agents/` | R6: LangGraph agents, explainer, critic |
| `ui/` | Streamlit app |
| `eval/` | metrics, `stats.py`, results |
| `config.yaml` | model routes, seeds, stats settings |
