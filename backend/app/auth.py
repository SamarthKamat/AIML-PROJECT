"""Authentication: registration, login (JWT HS256) and the @login_required decorator."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import insert, select
from werkzeug.security import check_password_hash, generate_password_hash

from backend.app.db import get_engine, users

bp = Blueprint("auth", __name__, url_prefix="/api/auth")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _token(user_id, email, role):
    payload = dict(sub=str(user_id), email=email, role=role, iat=datetime.now(timezone.utc),
                   exp=datetime.now(timezone.utc) + timedelta(hours=12))
    return jwt.encode(payload, current_app.config["SECRET_KEY"], algorithm="HS256")


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        if current_app.config.get("AUTH_DISABLED"):
            g.user = dict(user_id=None, email="anonymous", role="analyst")
            return fn(*a, **k)
        h = request.headers.get("Authorization", "")
        if not h.startswith("Bearer "):
            return jsonify(error="Unauthorized", message="missing bearer token"), 401
        try:
            p = jwt.decode(h[7:], current_app.config["SECRET_KEY"], algorithms=["HS256"])
        except jwt.PyJWTError as e:
            return jsonify(error="Unauthorized", message=str(e)), 401
        g.user = dict(user_id=int(p["sub"]), email=p["email"], role=p.get("role", "analyst"))
        return fn(*a, **k)
    return wrapper


@bp.post("/register")
def register():
    """Register a user: {email, password (>= 8 chars), full_name}."""
    d = request.get_json(silent=True) or {}
    email = str(d.get("email", "")).strip().lower()
    pw = str(d.get("password", ""))
    name = str(d.get("full_name", "")).strip()[:120]
    if not EMAIL_RE.match(email):
        return jsonify(error="Bad Request", message="invalid email"), 400
    if len(pw) < 8:
        return jsonify(error="Bad Request", message="password must be at least 8 characters"), 400
    with get_engine().begin() as c:
        if c.execute(select(users.c.user_id).where(users.c.email == email)).first():
            return jsonify(error="Conflict", message="email already registered"), 409
        r = c.execute(insert(users).values(email=email, full_name=name, password_hash=generate_password_hash(pw), role="analyst"))
        uid = r.inserted_primary_key[0]
    return jsonify(token=_token(uid, email, "analyst"), user=dict(user_id=uid, email=email, full_name=name)), 201


@bp.post("/login")
def login():
    """Log in: {email, password} -> JWT valid for 12 h."""
    d = request.get_json(silent=True) or {}
    email = str(d.get("email", "")).strip().lower()
    with get_engine().connect() as c:
        u = c.execute(select(users).where(users.c.email == email)).mappings().first()
    if not u or not check_password_hash(u["password_hash"], str(d.get("password", ""))):
        return jsonify(error="Unauthorized", message="invalid credentials"), 401
    return jsonify(token=_token(u["user_id"], u["email"], u["role"]),
                   user=dict(user_id=u["user_id"], email=u["email"], full_name=u["full_name"]))


@bp.get("/me")
@login_required
def me():
    """Current user from token."""
    return jsonify(g.user)
