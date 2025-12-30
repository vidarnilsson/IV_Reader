import cv2
import os
import pytesseract
import numpy as np

# x=70 y=2540

def process_image(image_bytes):

    # if not image_bytes:
    #     return jsonify({"error": "empty body"}), 400
# 
    # Decode raw bytes -> OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # img = cv2.imread("ho.jpg")  # For testing purposes only

    if img is None:
        print("Could not decode image.")
        #return jsonify({"error": "could not decode image"}), 400
    
    # Convert the image to grayscale (improves OCR accuracy)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the coordinates for cropping
    x1, y1 = 0, 2500  # top-left corner of the crop
    x2, y2 = 1320, 2868  # bottom-right corner of the crop
    cropped_text = gray[y1:y2, x1:x2]


    extracted_text = pytesseract.image_to_string(cropped_text)

    split_text_on_spaces = extracted_text.split()
    this_idx = split_text_on_spaces.index("This")
    pokemon_name = split_text_on_spaces[this_idx + 1]
    print(pokemon_name)

    x, y = 70, 2540       # Reference coordinates to check IV position
    pixel_value = int(gray[y, x])
    # IV crop coordinates
    if pixel_value > 200:
        y_offset = 84
    else:
        y_offset = 0
    x1, y1 = 150, 2130 - y_offset
    x2, y2 = 620, 2485 - y_offset
    cropped_iv = gray[y1:y2, x1:x2]

    x0, y0 = 13, 75
    ivs = []
    
    for i in range(3):
        orange_pixels = 0
        total_pixels = 0
        for x_value in range(x0, 453):
            pixel_value = int(cropped_iv[y0, x_value])
            if pixel_value < 200:
                orange_pixels += 1
            total_pixels += 1

        iv = round((orange_pixels / total_pixels) * 15)
        ivs.append(iv)
        y0 += 125  # Move to the next IV row
        
    
    cv2.imwrite('cropped_iv.jpg', cropped_iv)

    return pokemon_name, ivs

