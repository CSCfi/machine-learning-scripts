import glob
import json
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.engine.saving import load_model
from PIL import Image
from skimage import transform
from werkzeug.debug import DebuggedApplication
from werkzeug.wrappers import Response, Request
import numpy as np

"""
Development Usage:
    $ python valohai/prediction_server.py
    - it assumes that you have *.h5 model file in the current working directory
    - then you can POST images to the local URL e.g. with test_prediction_server.py
"""

model = None


def read_image_from_request(request):
    # Reads the first file in the request and tries to load it as an image.
    if not request.files:
        return None
    file_key = list(request.files.keys())[0]
    file = request.files.get(file_key)
    img = Image.open(file.stream)
    img.load()
    return img


def create_response(content, status_code):
    return Response(json.dumps(content), status_code, mimetype='application/json')


def predict_wsgi(environ, start_response):
    request = Request(environ)
    image = read_image_from_request(request)
    if not image:
        result = {'error': 'No images in the request, include sample image in the request.'}
        response = create_response(result, 400)
        return response(environ, start_response)

    # Pre-processing a single image
    # TODO: notice that this is not 100% the same preprocessing than in training, will create some skew
    image = np.array(image).astype('float32') / 255
    image = transform.resize(image, (150, 150, 3), mode='constant', anti_aliasing=False)
    image = np.expand_dims(image, axis=0)

    # Load model as global object so it stays in the memory making responses fast.
    global model
    if not model:
        # Try to find HDF5 files on the current directory to load as the model.
        local_hdf5_files = glob.glob('*.h5')
        if not local_hdf5_files:
            result = {'error': 'Could not find predictive model to load, contact support.'}
            response = create_response(result, 400)
            return response(environ, start_response)
        model_path = os.path.join(os.getcwd(), local_hdf5_files[0])
        model = load_model(model_path)

    # Give prediction on the image.
    predictions = model.predict(image)
    prediction = predictions[0]

    # Report results.
    cls = 'cat' if prediction < 0.5 else 'dog'
    result = {'class': cls, 'value': float(prediction)}
    response = create_response(result, 200)
    return response(environ, start_response)


predict_wsgi = DebuggedApplication(predict_wsgi)

if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple('localhost', 8000, predict_wsgi)
