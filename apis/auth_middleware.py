"""
Supabase JWT authentication middleware for Flask.

Usage:
    from .auth_middleware import require_auth

    @bp.route('/protected')
    @require_auth
    def protected_endpoint():
        user = g.user  # decoded JWT payload
        return jsonify({'user_id': user['sub']})

Requires environment variable:
    SUPABASE_JWT_SECRET  — the JWT secret from Supabase Dashboard > Settings > API
"""

from __future__ import annotations

import functools
import os

from flask import g, jsonify, request

try:
    import jwt as pyjwt  # PyJWT
except ModuleNotFoundError:
    pyjwt = None  # type: ignore[assignment]

SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")


def _verify_token(token: str) -> dict | None:
    """Decode and verify a Supabase access token. Returns payload or None."""
    if not pyjwt or not SUPABASE_JWT_SECRET:
        return None
    try:
        return pyjwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError):
        return None


def require_auth(f):
    """Decorator that enforces a valid Supabase Bearer token on a route."""

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not pyjwt or not SUPABASE_JWT_SECRET:
            # Auth not configured — pass through so existing dev workflow is unaffected.
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        payload = _verify_token(auth_header[7:])
        if payload is None:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user = payload
        return f(*args, **kwargs)

    return decorated


def init_auth(app):
    """
    Register an optional global before-request hook.
    Set REQUIRE_AUTH=1 in the environment to enforce auth on every request.
    """
    if not os.environ.get("REQUIRE_AUTH"):
        return

    @app.before_request
    def _global_auth_check():
        if request.method == "OPTIONS":
            return None
        if request.path == "/" or request.path.startswith("/health"):
            return None

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authentication required"}), 401

        payload = _verify_token(auth_header[7:])
        if payload is None:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user = payload
        return None
