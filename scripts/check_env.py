"""Check that this machine meets the project's requirements.

Run from the repo root:  python scripts/check_env.py
"""

import importlib.util
import json
import os
import platform
import shutil
import sys
import urllib.request

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def check_python():
    v = sys.version_info
    report(OK if (v.major, v.minor) >= (3, 11) else FAIL, "Python >= 3.11", platform.python_version())


def check_hardware():
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        # 7-8B Q4 models need ~5-6 GB; Neo4j container ~3 GB.
        report(OK if mem_gb >= 16 else WARN, "RAM >= 16 GB", f"{mem_gb:.0f} GB")
    except (ValueError, OSError):
        report(WARN, "RAM", "could not read")
    free_gb = shutil.disk_usage(".").free / 1024**3
    report(OK if free_gb >= 50 else WARN, "Free disk >= 50 GB", f"{free_gb:.0f} GB")


def check_packages():
    pkgs = [
        "pydantic", "jsonschema", "yaml", "dotenv", "numpy", "pandas", "datasketch", "pymupdf", "docx",
        "spacy", "sklearn_crfsuite", "sklearn", "rank_bm25", "neo4j", "networkx", "torch",
        "sentence_transformers", "faiss", "lightgbm", "ranx", "openai", "langgraph", "tenacity",
        "scipy", "statsmodels", "streamlit", "kaggle", "datasets",
    ]
    # find_spec checks presence without importing (importing kaggle triggers an auth prompt)
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    report(OK if not missing else FAIL, "Python packages", "missing: " + ", ".join(missing) if missing else f"{len(pkgs)} importable")


def check_torch_device():
    try:
        import torch
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        report(OK if dev != "cpu" else WARN, "Torch accelerator", dev)
    except ImportError:
        report(FAIL, "Torch accelerator", "torch not installed")


def check_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        report(OK, "spaCy model en_core_web_sm")
    except Exception as e:
        report(FAIL, "spaCy model en_core_web_sm", type(e).__name__)


def check_ollama():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.load(r)["models"]]
        report(OK if models else WARN, "Ollama server", ", ".join(models) or "running, no models pulled")
    except Exception as e:
        report(FAIL, "Ollama server", f"not reachable ({type(e).__name__})")


def check_neo4j():
    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        load_dotenv()
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as s:
                version = s.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
                try:
                    gds = s.run("RETURN gds.version() AS v").single()["v"]
                except Exception:
                    gds = None
        report(OK, "Neo4j", f"v{version}" + (f", GDS {gds}" if gds else ""))
        if not gds:
            report(WARN, "Neo4j GDS plugin", "not loaded (needed for Louvain in-graph)")
    except Exception as e:
        report(FAIL, "Neo4j", f"{type(e).__name__}: {e}"[:120])


def check_api_keys():
    from dotenv import load_dotenv
    load_dotenv()
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        report(OK if os.getenv(key) else WARN, key, "set" if os.getenv(key) else "not set in .env")
    kaggle_files = ("~/.kaggle/kaggle.json", "~/.kaggle/access_token", "~/.kaggle/credentials.json")
    kaggle_ok = (any(os.path.exists(os.path.expanduser(f)) for f in kaggle_files)
                 or os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN"))
    report(OK if kaggle_ok else WARN, "Kaggle credentials", "found" if kaggle_ok else "missing (run `kaggle auth login`)")


if __name__ == "__main__":
    for check in (check_python, check_hardware, check_packages, check_torch_device,
                  check_spacy_model, check_ollama, check_neo4j, check_api_keys):
        check()
    print(f"\n{results.count(OK)} ok, {results.count(WARN)} warnings, {results.count(FAIL)} failures")
    sys.exit(1 if FAIL in results else 0)
