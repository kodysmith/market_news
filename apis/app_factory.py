from __future__ import annotations

from flask import Flask
try:
    from flask_cors import CORS  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    CORS = None


def create_app() -> Flask:
    app = Flask(__name__)
    # This will allow all domains. For production, restrict origins!
    if CORS is not None:
        CORS(app)

    # Optional global auth enforcement (set REQUIRE_AUTH=1 to enable)
    from .auth_middleware import init_auth
    init_auth(app)

    # Register modular blueprints
    from .routes_report import bp as report_bp
    from .routes_news import bp as news_bp
    from .routes_quantbot import bp as quantbot_bp
    from .routes_economic_calendar import bp as econ_bp
    from .routes_gex import bp as gex_bp
    from .routes_valuation import bp as valuation_bp
    from .routes_cockpit import bp as cockpit_bp
    from .routes_depin import bp as depin_bp
    from .routes_probability import bp as probability_bp
    from .routes_fisher import bp as fisher_bp
    from .routes_index import bp as index_bp
    from .routes_scanner import bp as scanner_bp
    from .routes_bot import bp as bot_bp

    app.register_blueprint(report_bp)
    app.register_blueprint(news_bp)
    app.register_blueprint(quantbot_bp)
    app.register_blueprint(econ_bp)
    app.register_blueprint(gex_bp)
    app.register_blueprint(valuation_bp)
    app.register_blueprint(cockpit_bp)
    app.register_blueprint(depin_bp)
    app.register_blueprint(probability_bp)
    app.register_blueprint(fisher_bp)
    app.register_blueprint(index_bp)
    app.register_blueprint(scanner_bp)
    app.register_blueprint(bot_bp)

    return app

