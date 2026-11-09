"""Cloud Run service for the StegoGuard-LM results explorer.

Serves the built explorer page and the results bundle it was built from.
Both files are baked into the container image by scripts/deploy_explorer.sh,
so the service reads nothing from the research data at run time.

Routes:
    /                    the explorer page
    /api/bundle          the results bundle as JSON (for a future dynamic client)
    /healthz             liveness check

Added after the experiments; not covered by the freeze manifests.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, abort, send_from_directory

DATA_DIR = Path(__file__).resolve().parent / "data"

app = Flask(__name__)


@app.after_request
def headers(resp: Response) -> Response:
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


@app.get("/")
def index() -> Response:
    return send_from_directory(DATA_DIR, "index.html", mimetype="text/html")


@app.get("/api/bundle")
def bundle() -> Response:
    if not (DATA_DIR / "explorer_bundle.json").is_file():
        abort(404)
    return send_from_directory(DATA_DIR, "explorer_bundle.json", mimetype="application/json")


@app.get("/healthz")
def healthz() -> tuple[str, int]:
    return "ok", 200
