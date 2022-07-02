import numpy as np
from flask import abort
from paddle_serving_app.reader import *
from paddle_serving_server.web_service import WebService

preprocess = Sequential([
    Base64ToImage(), Normalize((123,117,104), (127.502231, 127.502231, 127.502231)), Transpose((2, 0, 1)),
])


class FaceWebService(WebService):

    def get_prediction(self, request):
        if not request.json:
            abort(400)
        if request.json["feed"]["threshold"] < 0 or request.json["feed"]["threshold"] > 1:
            abort(400)
        try:
            feed, fetch, is_batch = self.preprocess(request.json["feed"],
                                                    ["multiclass_nms3_0.tmp_0"])
            if isinstance(feed, dict) and "fetch" in feed:
                del feed["fetch"]
            if len(feed) == 0:
                raise ValueError("empty input")
            fetch_map = self.client.predict(
                feed=feed, fetch=fetch, batch=is_batch)
            result = self.postprocess(
                feed=request.json["feed"], fetch=fetch, fetch_map=fetch_map)
            result = {"result": result}
        except ValueError as err:
            result = {"result": str(err)}
        return result

    def preprocess(self, feed={}, fetch=[]):
        is_batch = False
        img = preprocess(feed["image"])
        feed["image"] = img
        feed["im_shape"] = np.array(list(img.shape[1:])).reshape(-1)
        feed["scale_factor"] = np.array([1.0, 1.0]).reshape(-1)
        return feed, fetch, is_batch

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        for key in fetch_map:
            fetch_map[key] = fetch_map[key].tolist()
        threshold = feed['threshold'] if feed['threshold'] else 0.9
        fetch_map['faces'] = [arr[1:] for arr in fetch_map['multiclass_nms3_0.tmp_0'] if arr[1] > threshold]
        size = feed['size'] if feed['size'] else len(fetch_map['face'])
        fetch_map['faces'] = fetch_map['faces'][0:size]
        fetch_map['size'] = len(fetch_map['faces'])
        del fetch_map['multiclass_nms3_0.tmp_0']
        del fetch_map['multiclass_nms3_0.tmp_0.lod']
        return fetch_map


if __name__ == '__main__':
    face_service = FaceWebService(name="FaceDetection")
    face_service.load_model_config("model")
    face_service.prepare_server(workdir="workdir", port=9393)
    face_service.run_debugger_service()
    face_service.run_web_service()