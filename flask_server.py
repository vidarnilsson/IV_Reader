from flask import Flask, request, jsonify
from img_processor import process_image
from calc_iv_ranks import get_all_ranks
import os
import cv2
import numpy as np


app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    print("Content-Type:", request.content_type)
    print("request.files:", request.files)

    try:
        image_bytes = request.get_data()
        pokemon_name, ivs = process_image(image_bytes)
        final_string = get_all_ranks(pokemon_name, ivs)

    except Exception as e:
        return "Oops, something went wrong"

    return final_string

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)