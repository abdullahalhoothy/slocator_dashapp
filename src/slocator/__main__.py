"""Entrypoint: ``python -m slocator``."""

from .app import app
from .config import Config


def main() -> None:
    app.run(host=Config.APP_HOST, port=Config.APP_PORT, debug=Config.APP_DEBUG)


if __name__ == "__main__":
    main()