import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import cv2
from flask import Flask, jsonify, request

from calc_iv_ranks import get_all_ranks
from img_processor import process_image

app = Flask(__name__)

# Configure logging
LOG_FILE = "upload_log.txt"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5  # Keep 5 backup files

# Create a custom logger
logger = logging.getLogger("upload_logger")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT
)
file_handler.setLevel(logging.INFO)

# Create a custom format
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)


def log_request(pokemon_name=None, ivs=None, response=None, error=None):
    """Log details about each upload request."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get client IP address
    if request.headers.get("X-Forwarded-For"):
        client_ip = request.headers.get("X-Forwarded-For").split(",")[0].strip()
    else:
        client_ip = request.remote_addr or "unknown"

    # Get content info
    content_length = request.content_length or 0
    content_type = request.content_type or "unknown"
    user_agent = request.headers.get("User-Agent", "unknown")

    # Build log entry
    log_parts = [
        f"[{timestamp}]",
        f"IP: {client_ip}",
        f"Size: {content_length} bytes",
        f"Content-Type: {content_type}",
    ]

    if pokemon_name and ivs:
        log_parts.append(f"Pokemon: {pokemon_name}")
        log_parts.append(f"IVs: {ivs[0]}/{ivs[1]}/{ivs[2]}")

    if response:
        # Truncate long responses for the log
        response_preview = response[:100] + "..." if len(response) > 100 else response
        response_preview = response_preview.replace("\n", " ")
        log_parts.append(f"Response: {response_preview}")

    if error:
        log_parts.append(f"Error: {error}")

    log_parts.append(f"User-Agent: {user_agent}")

    log_entry = " | ".join(log_parts)
    logger.info(log_entry)


@app.route("/upload", methods=["POST"])
def upload():
    print("Content-Type:", request.content_type)
    print("request.files:", request.files)

    pokemon_name = None
    ivs = None
    final_string = None
    error_msg = None

    try:
        image_bytes = request.get_data()
        pokemon_name, ivs = process_image(image_bytes)
        final_string = get_all_ranks(pokemon_name, ivs)

        # Log successful request
        log_request(pokemon_name=pokemon_name, ivs=ivs, response=final_string)

        return final_string

    except Exception as e:
        error_msg = str(e)

        # Log failed request
        log_request(pokemon_name=pokemon_name, ivs=ivs, error=error_msg)

        return "Oops, something went wrong"


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
