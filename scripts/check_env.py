"""PHASE 0 GATE — verify every external credential actually works.

Each check makes a real network call. Nothing in this project proceeds until
every required check passes.

    python scripts/check_env.py

Secrets are never printed; only a masked fingerprint (first 6 / last 4 chars)
so you can tell which key is loaded without exposing it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

try:
    import httpx
    from dotenv import load_dotenv
except ImportError:
    sys.exit(
        "Missing base deps. Run:\n"
        "  uv venv --python 3.11 && uv pip install httpx python-dotenv "
        "qdrant-client neo4j"
    )

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

G, R, Y, B, DIM, RST = (
    "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[0m"
)
TIMEOUT = 20.0


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    warn: bool = False
    notes: list[str] = field(default_factory=list)


def mask(v: str | None) -> str:
    if not v:
        return f"{DIM}(unset){RST}"
    return f"{v[:6]}…{v[-4:]}" if len(v) > 14 else f"{v[:3]}…"


def need(*keys: str) -> tuple[bool, str]:
    missing = [k for k in keys if not os.getenv(k, "").strip()]
    return (not missing), ("missing: " + ", ".join(missing) if missing else "")


# --------------------------------------------------------------------------
# 1. Gemini
# --------------------------------------------------------------------------
def check_gemini() -> Result:
    fix = (
        "Create a Gemini API key at https://aistudio.google.com/apikey and set "
        "GEMINI_API_KEY (or OPENAI_API_KEY) in .env"
    )
    key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not key.strip():
        return Result("Gemini", False, "missing: GEMINI_API_KEY or OPENAI_API_KEY", fix)

    base = (
        os.getenv("GEMINI_BASE_URL")
        or os.getenv(
            "OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    )
    fast = os.getenv("LLM_FAST_MODEL", "gemini-3.8-flash")
    smart = os.getenv("LLM_SMART_MODEL", "gemini-pro-latest")
    notes: list[str] = []

    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            catalog = c.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": key},
            )
            if catalog.status_code in (400, 401, 403):
                return Result(
                    "Gemini", False,
                    f"{catalog.status_code} from Gemini model catalog; check the key and API access",
                    fix,
                )
            catalog.raise_for_status()
            available = {
                item["name"].rsplit("/", 1)[-1]
                for item in catalog.json().get("models", [])
            }

            for model in (fast, smart):
                if model not in available:
                    return Result("Gemini", False, f"model not available: {model}", fix)

            headers = {"Authorization": "Bearer " + key}
            endpoint = f"{base.rstrip('/')}/chat/completions"
            completion = c.post(
                endpoint,
                headers=headers,
                json={
                    "model": fast,
                    "messages": [{"role": "user", "content": "Reply with: ok"}],
                    "max_tokens": 8,
                },
            )
            if completion.status_code == 429:
                return Result(
                    "Gemini", False,
                    "429 from Gemini generation; check quota and billing in Google AI Studio",
                    "Check https://aistudio.google.com and the Gemini API rate limits",
                )
            completion.raise_for_status()

            tool_probe = c.post(
                endpoint,
                headers=headers,
                json={
                    "model": fast,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "stream": False,
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }],
                },
            )
            if tool_probe.status_code != 200:
                notes.append(
                    f"{Y}tool-calling probe failed ({tool_probe.status_code}) - "
                    f"verify NeMo compatibility before deploying{RST}"
                )
            elif not tool_probe.json()["choices"][0]["message"].get("tool_calls"):
                notes.append(
                    f"{Y}tool-calling returned no tool_calls - verify before deployment{RST}"
                )
            else:
                notes.append(f"{DIM}tool-calling OK (NeMo-compatible){RST}")

    except httpx.HTTPError as e:
        return Result("Gemini", False, f"{type(e).__name__}: {e}", fix)

    return Result(
        "Gemini", True, f"{mask(key)} - {fast} + {smart} reachable", notes=notes
    )


# --------------------------------------------------------------------------
# 2. Qdrant Cloud
# --------------------------------------------------------------------------
def check_qdrant() -> Result:
    fix = "https://cloud.qdrant.io -> create free cluster -> API Keys; set QDRANT_URL + QDRANT_API_KEY"
    ok, why = need("QDRANT_URL", "QDRANT_API_KEY")
    if not ok:
        return Result("Qdrant", False, why, fix)
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return Result("Qdrant", False, "qdrant-client not installed",
                      "uv pip install qdrant-client")
    try:
        client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            timeout=int(TIMEOUT),
        )
        names = [c.name for c in client.get_collections().collections]
    except Exception as e:
        return Result("Qdrant", False, f"{type(e).__name__}: {e}", fix)

    existing = f"{len(names)} collection(s): {', '.join(names)}" if names else "no collections yet"
    return Result("Qdrant", True, f"connected - {existing}")


# --------------------------------------------------------------------------
# 3. Langfuse Cloud
# --------------------------------------------------------------------------
def check_langfuse() -> Result:
    fix = "https://cloud.langfuse.com -> project -> Settings -> API Keys"
    ok, why = need("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    if not ok:
        return Result("Langfuse", False, why, fix)

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    pk, sk = os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.get(f"{host}/api/public/health", auth=(pk, sk))
            if r.status_code == 401:
                return Result(
                    "Langfuse", False,
                    "401 - keys rejected (check they're from the same project, "
                    "and that LANGFUSE_HOST matches your region)", fix,
                )
            r.raise_for_status()
    except httpx.HTTPError as e:
        return Result("Langfuse", False, f"{type(e).__name__}: {e}", fix)
    return Result("Langfuse", True, f"{mask(pk)} @ {host}")


# --------------------------------------------------------------------------
# 4. Tavily
# --------------------------------------------------------------------------
def check_tavily() -> Result:
    fix = "https://tavily.com -> sign up -> dashboard -> copy key (tvly-...)"
    ok, why = need("TAVILY_API_KEY")
    if not ok:
        return Result("Tavily", False, why, fix)
    key = os.environ["TAVILY_API_KEY"]
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": "neo4j", "max_results": 1},
            )
            if r.status_code in (401, 403):
                return Result("Tavily", False, f"{r.status_code} - key rejected", fix)
            if r.status_code == 429:
                return Result("Tavily", False, "429 - monthly quota exhausted",
                              "Wait for reset or upgrade at https://tavily.com")
            r.raise_for_status()
            n = len(r.json().get("results", []))
    except httpx.HTTPError as e:
        return Result("Tavily", False, f"{type(e).__name__}: {e}", fix)
    return Result("Tavily", True, f"{mask(key)} - search returned {n} result(s)")


# --------------------------------------------------------------------------
# 5. Neo4j (local Docker)
# --------------------------------------------------------------------------
def check_neo4j() -> Result:
    fix = "Set NEO4J_PASSWORD in .env, then: make up"
    ok, why = need("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    if not ok:
        return Result("Neo4j", False, why, fix)
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return Result("Neo4j", False, "neo4j driver not installed",
                      "uv pip install neo4j")

    uri = os.environ["NEO4J_URI"]
    try:
        driver = GraphDatabase.driver(
            uri, auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        )
        driver.verify_connectivity()
        with driver.session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        driver.close()
    except Exception as e:
        msg = str(e)
        hint = fix
        if "authentication" in msg.lower() or "unauthorized" in msg.lower():
            hint = ("Password mismatch between .env and the container's stored auth. "
                    "Reset with: docker compose down -v && make up")
        elif "could not connect" in msg.lower() or "refused" in msg.lower():
            hint = f"Nothing listening on {uri}. Start it with: make up"
        return Result("Neo4j", False, f"{type(e).__name__}: {msg[:110]}", hint)

    state = f"{nodes} nodes (graph built)" if nodes else "empty (expected before `make ingest`)"
    return Result("Neo4j", True, f"{uri} - {state}")


CHECKS = [check_gemini, check_qdrant, check_langfuse, check_tavily, check_neo4j]


def main() -> int:
    if not os.path.exists(os.path.join(REPO_ROOT, ".env")):
        print(f"\n{R}No .env file found.{RST}\n\n  cp .env.example .env\n\n"
              "Then fill in the values described in .env.example.\n")
        return 1

    print(f"\n{B}PHASE 0 - credential and connectivity check{RST}\n")
    results = []
    for fn in CHECKS:
        try:
            res = fn()
        except Exception as e:  # a check must never crash the gate
            res = Result(fn.__name__, False, f"checker crashed: {type(e).__name__}: {e}")
        results.append(res)
        icon = f"{G}PASS{RST}" if res.ok else (f"{Y}WARN{RST}" if res.warn else f"{R}FAIL{RST}")
        print(f"  [{icon}] {res.name:<11} {res.detail}")
        for n in res.notes:
            print(f"           {DIM}|{RST} {n}")
        if not res.ok and res.fix:
            print(f"           {DIM}->{RST} {res.fix}")

    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"{R}{len(failed)} of {len(results)} checks failed.{RST} "
              "Phase 1 is blocked until these pass.\n")
        return 1
    print(f"{G}All {len(results)} checks passed.{RST} Phase 0 complete - ready for Phase 1.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
