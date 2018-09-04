# -*- coding: utf-8 -*-
from bottle import route, run, template, request, redirect
import os
import cv2
import sys
import os.path
import tempfile
import base64
import requests
import json
import numpy as np

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


API_KEY = "AIzaSyCJ-a12ftbkW1silfpbElVC0Du8hL20JRk"
def text_detection(image_path):
    api_url = 'https://vision.googleapis.com/v1/images:annotate?key={}'.format(API_KEY)
    with open(image_path, "rb") as img:
        image_content = base64.b64encode(img.read())
        req_body = json.dumps({
            'requests': [{
                'image': {
                    'content': image_content.decode('utf-8')  # base64でエンコードしたものjsonにするためdecodeする
                },
                'features': [{
                    'type': 'TEXT_DETECTION'
                }]
            }]
        })
        res = requests.post(api_url, data=req_body)
        return res.json()




ocr = cv2.text.OCRTesseract_create()
original = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABUUlEQVQ4T43TP0iVURzG8c8VJEEQpIZ0CdRB1HAI3Ezc2qzBSREXkYiCcGiKahDxDyJIDSrh2qCbi+DsIA4uOuiiFBI4BBfuZsSRcy5v13vve9/tnN/vfM/vec7zFlT/evAHNzXq5e1ClYYZPEQH9rBfD1IJGMYzrMVDy1jHFdrxAd/i+q6lEjCNMxzhMYKUv/iN12jGCi7TVAnwBJ04xksMognneIQHWMB3TGUlJcAG3uINfuFHDd0DGMen7AT9mIi6TnCY4/wL9Caf0gQj+IzRnMPh5k28QiuWEqAPY1FnPcZHbOE6elVKgKd4jq95wcnUwxTFBAiOL0ZzSg1AgtnhZVazOejGJL7kAOaxHZ/4XpBCBoo4wFAMULjpJ8Jks7F2URmktJ6LSQwOd+EWLWiLEt/FjJzWAoT99zGquxkpIY0hyjsoH672LzTg3/8t/wCs6Tqb27/Z5AAAAABJRU5ErkJggg=='
result = original
list = []

def order(x,y):
	if x > y:
		return y, x
	else:
		return x, y

def tesseract(img, x, y, w, h):
	ret, mask = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY) # make black mask
	print(x ,y ,w, h)
	x_min, x_max = order(x, w)
	for i in range(x_min, x_max):
		y_min, y_max = order(y, h)
		for j in range(y_min, y_max):
			mask[j][i] = 0
	ret = ocr.run(img, mask, 1)
	tmp_img = img[y:y+h, x:x+w]
	return ocr.run(tmp_img, 1)

def captch_ex(file):

    # アップロードされたファイルを一時データとして保存する
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file.read())
        file_name = tf.name


    # cloud vision api
    result_json = text_detection(file_name)

    original = cv2.imread(file_name)
    result = original.copy()

    logger.debug(result_json)

    for d in result_json["responses"][0]["textAnnotations"][1:]:
        logger.debug(d["boundingPoly"]["vertices"])
        vertices = d["boundingPoly"]["vertices"]
        pts = np.array([[e["x"],e["y"]] for e in vertices], np.int32)
        pts = pts.reshape((-1,1,2))
        logger.debug(vertices)
        result = cv2.polylines(result,[pts],True,(255,100,100), 3)



    cv2.imwrite('/usr/local/src/original.png', original)
    cv2.imwrite('/usr/local/src/result.png', result)

    return original, result

@route('/')
def index():
    return template('form', original=original, result=result, list=list)


@route('/upload', method='POST')
def do_upload():
    global list
    list = []
    upload = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)
    print(upload.filename);

    original, result = captch_ex(upload.file)    

    with tempfile.NamedTemporaryFile() as tf:

        cv2.imwrite(tf.name + '_origin.png', original)
        cv2.imwrite(tf.name + '_result.png', result)

        with open(tf.name + '_origin.png', "rb") as imgfile:
            data = imgfile.read()
            global original
            original = 'data:image/png;base64,' + base64.encodestring(data).decode('utf8')
        with open(tf.name + '_result.png', "rb") as imgfile:
            data = imgfile.read()
            global result
            result = 'data:image/png;base64,' + base64.encodestring(data).decode('utf8')

    redirect('/')

run(host='0.0.0.0', port=8080, debug=True, reloader=True)
