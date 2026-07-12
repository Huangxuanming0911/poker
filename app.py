"""Zeabur-friendly Flask entrypoint.

The main app lives in server.py. Zeabur detects app.py/main.py for Python
services, so this file exposes the same Flask app and supports local startup
with the platform-provided PORT environment variable.
"""

import os

from server import app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
