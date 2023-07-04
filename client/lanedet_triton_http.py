import cv2
import numpy as np
import sys
import time
import tritonclient.http as httpclient
import gevent.ssl
import matplotlib.pyplot as plt


def client_init(url="10.11.18.117:9000",
                ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False,
                verbose=False):
    """

    :param url:
    :param ssl: Enable encrypted link to the server using HTTPS
    :param key_file: File holding client private key
    :param cert_file: File holding client certificate
    :param ca_certs: File holding ca certificate
    :param insecure: Use no peer verification in SSL communications. Use with caution
    :param verbose: Enable verbose output
    :return:
    """
    if ssl:
        ssl_options = {}
        if key_file is not None:
            ssl_options['keyfile'] = key_file
        if cert_file is not None:
            ssl_options['certfile'] = cert_file
        if ca_certs is not None:
            ssl_options['ca_certs'] = ca_certs
        ssl_context_factory = None
        if insecure:
            ssl_context_factory = gevent.ssl._create_unverified_context
        triton_client = httpclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=True,
            ssl_options=ssl_options,
            insecure=insecure,
            ssl_context_factory=ssl_context_factory)
    else:
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=verbose)

    return triton_client



class LaneDetClient():
    def __init__(self):
        self.triton_client = client_init()
        self.inputs = []
        self.outputs = []
        self.inputs.append(httpclient.InferInput('input', [1, 3, 360, 640], 'FP32'))
        self.outputs.append(httpclient.InferRequestedOutput('predict_lanes', binary_data=True))
        self.outputs.append(httpclient.InferRequestedOutput('914', binary_data=True))
        self.inputs[0].set_data_from_numpy(np.random.random([1, 3, 360, 640]).astype(np.float32), binary_data=True)
        self.model_name = 'LaneDet'
    
    def __call__(self, input_tensor):
        self.inputs[0].set_data_from_numpy(input_tensor)
        result = self.triton_client.infer(self.model_name, inputs=self.inputs, outputs=self.outputs)
        output = result.as_numpy("predict_lanes")
        return output

class PreprocessClient():
    def __init__(self):
        self.triton_client = client_init()
        self.inputs = []
        self.outputs = []
        self.inputs.append(httpclient.InferInput('input_img', [1, 1080, 1920, 3], 'UINT8'))
        self.outputs.append(httpclient.InferRequestedOutput('output_tensor'))
        self.inputs[0].set_data_from_numpy(np.random.random([1, 1080, 1920, 3]).astype(np.uint8))
        self.model_name = 'preprocess'
    
    def __call__(self, img):
        self.inputs[0].set_data_from_numpy(img)
        result = self.triton_client.infer(self.model_name, inputs=self.inputs, outputs=self.outputs)
        output = result.as_numpy("output_tensor")
        return output

class PostprocessClient():
    def __init__(self):
        self.triton_client = client_init()
        self.inputs = []
        self.outputs = []
        self.inputs.append(httpclient.InferInput('prediction', [1, 2784, 77], 'FP32'))
        self.outputs.append(httpclient.InferRequestedOutput('lanes'))
        self.inputs[0].set_data_from_numpy(np.random.random([1, 2784, 77]).astype(np.float32))
        self.model_name = 'postprocess'
    
    def __call__(self, prediction):
        self.inputs[0].set_data_from_numpy(prediction)
        result = self.triton_client.infer(self.model_name, inputs=self.inputs, outputs=self.outputs)
        output = result.as_numpy("lanes")
        return output

class PiplineClient():
    def __init__(self):
        self.triton_client = client_init()
        self.inputs = []
        self.outputs = []
        self.inputs.append(httpclient.InferInput('LaneDet_input', [1, 1080, 1920, 3], 'UINT8'))
        self.outputs.append(httpclient.InferRequestedOutput('LaneDet_output'))
        self.inputs[0].set_data_from_numpy(np.random.random([1, 1080, 1920, 3]).astype(np.uint8))
        self.model_name = 'LaneDet_pipline'
    
    def __call__(self, img):
        self.inputs[0].set_data_from_numpy(img)
        result = self.triton_client.infer(self.model_name, inputs=self.inputs, outputs=self.outputs)
        output = result.as_numpy("LaneDet_output")
        return output

def showImg(img, y_samples, predictions):
    img_h, img_w, _ = img.shape
    y_samples = [i * img_h / 720 for i in y_samples]
    for line in predictions:
        line = [i * img_w / 1280 for i in line]
        cv2.polylines(img, np.int32([list(tups for tups in zip(line, y_samples) if tups[0] > 0 )]), isClosed=False, color=(0, 255, 0), thickness=2)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    client_init()
    img_path = 'D:\data\ca_front_left_image_jpg\\1.jpg'
    img = plt.imread(img_path)
    img_h, img_w, channel = img.shape
    print(img.shape)
    img_input = img[np.newaxis, :, :, :]
    preprocess = PiplineClient()
    for i in range(10):
        output = preprocess(img_input)
        print(i)
    # y_samples =  [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330,
    #      340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550,
    #       560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
    # showImg(img, y_samples, output[0])
    