import requests
import json
import base64  # 画像はbase64でエンコードする必要があるため

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


if __name__ == "__main__":
    img_path = "/usr/local/src/original.png"
    res_json = text_detection(img_path)
    res_text = res_json["responses"][0]["textAnnotations"][1]["boundingPoly"]
    #print(json.dumps(res_json, indent=4, sort_keys=True, ensure_ascii=False))
    print(res_text)
    with open("result.json", "w") as js:
        #json.dump(res_json, js, indent=4, ensure_ascii=False)
        js.write(res_text)
