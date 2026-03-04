"""
Optional global auth enforcement for the Flask app.

Set REQUIRE_AUTH=1 in the environment to enable bearer-token checks on every
request.  The expected token is read from the AUTH_TOKEN env var.
"""

from __future__ import annotations

import os

from flask import Flask, abort, request


def init_auth(app: Flask) -> None:
    if not os.getenv("REQUIRE_AUTH"):
        return

    expected_token = os.getenv("AUTH_TOKEN", "")

    @app.before_request
    def _check_auth() -> None:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != expected_token:
            abort(401)
