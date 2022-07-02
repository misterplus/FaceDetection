import requests
import json
import base64
import cv2 # 如果需要上传图片
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def cv2_to_base64(image):
    with open(image, 'rb') as img:
        return base64.b64encode(img.read()).decode('utf8')


if __name__ == '__main__':
    filename = "./test.jpg"
    img = cv2_to_base64(filename)
    headers = {"Content-type": "application/json"}
    url = "http://192.168.216.1:9393/FaceDetection/prediction"
    data = {
            'feed': {
                "image": img,
                'threshold': 0.9,
                'size': 3
            }
        }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    result = r.json()['result']
    print(result)

    mat = cv2.imread(filename)
    for face in result['faces']:
        x1, y1, x2, y2 = face[1:]
        cv2.rectangle(mat, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  ##根据坐标画框
    cv2.imwrite("output.jpg", mat)
    lena = mpimg.imread("output.jpg")
    plt.imshow(lena)  # 显示图片
    plt.axis('off')  # 不显示坐标轴

    plt.show()