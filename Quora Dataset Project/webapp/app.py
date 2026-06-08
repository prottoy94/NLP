import os

from controller.routes import app_routes
from flask import Flask


def create_app():
    app = Flask(__name__)
    app.register_blueprint(app_routes)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    create_app().run(host="127.0.0.1", port=port, debug=debug)
