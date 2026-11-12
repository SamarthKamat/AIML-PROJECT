"""Flask application factory for the Supply Chain Black Box API."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, g, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

API_VERSION = "1.0.0"


def _nan_safe(o):
    import math
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _nan_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_safe(v) for v in o]
    return o


def create_app(test_config: dict | None = None) -> Flask:
    dist = ROOT / "frontend" / "dist"
    app = Flask(__name__, static_folder=None)  # SPA served by the catch-all route below
    app.config.update(
        SECRET_KEY=os.environ.get("JWT_SECRET", "dev-only-insecure-secret-change-me-in-production-0123456789"),
        AS_OF=os.environ.get("APP_AS_OF", "2025-12-01"),
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY", ""),
        GEMINI_MODEL=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
        AUTH_DISABLED=os.environ.get("AUTH_DISABLED", "0") == "1",
    )
    if test_config:
        app.config.update(test_config)
    from flask.json.provider import DefaultJSONProvider

    class SafeJSON(DefaultJSONProvider):  # NaN/inf -> null so browsers always receive valid JSON
        def dumps(self, obj, **kw):
            return super().dumps(_nan_safe(obj), **kw)

    app.json = SafeJSON(app)
    CORS(app, resources={r"/api/*": {"origins": os.environ.get("CORS_ORIGINS", "*")}})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("scbb")

    from backend.app.db import get_engine, metadata
    metadata.create_all(get_engine())

    from backend.app.auth import bp as auth_bp
    from backend.app.routes_core import bp as core_bp
    from backend.app.routes_ml import bp as ml_bp
    from backend.app.assistant import bp as chat_bp
    for b in (auth_bp, core_bp, ml_bp, chat_bp):
        app.register_blueprint(b)

    @app.before_request
    def _start():
        g.t0 = time.time()

    @app.after_request
    def _log(resp):
        if request.path.startswith("/api"):
            log.info("%s %s %s %.0fms", request.method, request.path, resp.status_code, (time.time() - g.get("t0", time.time())) * 1000)
        return resp

    @app.errorhandler(Exception)
    def _err(e):
        if isinstance(e, HTTPException):
            return jsonify(error=e.name, message=e.description), e.code
        if isinstance(e, (KeyError, ValueError)):
            return jsonify(error="Bad Request", message=str(e)), 400
        log.exception("unhandled error")
        return jsonify(error="Internal Server Error", message=str(e)), 500

    @app.get("/api/health")
    def health():
        from sqlalchemy import text
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return jsonify(status="ok", version=API_VERSION, as_of=app.config["AS_OF"],
                       database=get_engine().dialect.name, gemini_configured=bool(app.config["GEMINI_API_KEY"]))

    @app.get("/api/docs")
    def docs():
        routes = sorted({(r.rule, ",".join(sorted(m for m in r.methods if m not in ("HEAD", "OPTIONS"))), r.endpoint)
                         for r in app.url_map.iter_rules() if r.rule.startswith("/api")})
        return jsonify([dict(path=p, methods=m, endpoint=e, doc=(app.view_functions[e].__doc__ or "").strip().split("\n")[0])
                        for p, m, e in routes])

    if dist.exists():
        @app.get("/")
        @app.get("/<path:path>")
        def spa(path="index.html"):
            if path.startswith("api/"):
                return jsonify(error="Not Found", message="unknown API route"), 404
            f = dist / path
            if path and f.exists() and f.is_file():
                return send_from_directory(dist, path)
            return send_from_directory(dist, "index.html")  # client-side routing

    return app
