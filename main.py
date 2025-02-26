from flask import Flask
from flask import request
from flask import json
from flask.wrappers import Request
from ocr import draw_boxes, parse, parse_to_image, image_boxes, image_boxes_to_text, image_boxes_to_text_vision_api, image_2dboxes_to_text, image_2dboxes_to_text_vision_api
from six.moves import urllib
import os
app = Flask(__name__)

from flask import send_file

import cv2
import numpy as np


@app.route('/')
def hello_world():
    return 'Python: OCR'

@app.route('/ocr')
def ocr():
    url = request.args.get("url")
    #url = urllib.parse.unquote(url)
    print(url)
    boxes = parse(url)
    print(boxes)
    return json.jsonify(boxes)

@app.route('/ocr-to-image')
def ocr2image():
    url = request.args.get("url")
    #url = urllib.parse.unquote(url)
    print(url)
    filename = parse_to_image(url)
    
    # Create the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Sharpen the image
    sharpened_image = cv2.filter2D(filename, -1, kernel)

    return send_file(sharpened_image,  mimetype='image/png')

@app.route('/image-rect', methods=['GET', 'POST'])
def imagerect():
    url = request.args.get("url")
    content = request.json
    filename = image_boxes(url, content["positions"])
    #url = urllib.parse.unquote(url)
    print(url)
    return send_file(filename,  mimetype='image/png')

# untuk form
@app.route('/imageboxes2text', methods=['GET', 'POST'])
def imageboxes2text():
    url = request.args.get("url")
    content = request.json

    print("imageboxes2text : Positions")
    print(content["positions"])
    
    #positions = image_boxes_to_text(url, content["positions"])
    positions = image_boxes_to_text_vision_api(url, content["positions"])
    return json.jsonify(positions)

# untuk table
@app.route('/image2dboxes2text', methods=['GET', 'POST'])
def image2dboxes2text():
    url = request.args.get("url")
    content = request.json

    print("/image2dboxes2text: Positions")
    print(content["positions"])
    
    # pakai tesseract
    positions = image_2dboxes_to_text(url, content["positions"])

    # pakai vision
    #positions = image_2dboxes_to_text_vision_api(url, content["positions"])
    
    return json.jsonify(positions)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=os.environ['APPLICATION_PORT'])


