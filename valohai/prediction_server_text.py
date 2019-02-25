import glob
import json
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing import sequence, text
from keras.engine.saving import load_model
from werkzeug.debug import DebuggedApplication
from werkzeug.wrappers import Response, Request
import gzip

"""
Development Usage:
    $ python valohai/prediction_server_text.py
    - it assumes that you have *.h5 model file in the current working directory
    - submit texts with GET parameter text to the local URL e.g. with test_prediction_server.py
"""

MAX_SEQUENCE_LENGTH = 1000

model = None
tokenizer = None

groups = {
    'atk': 0,
    'harrastus': 1,
    'keskustelu': 2,
    'misc': 3,
    'tiede': 4,
    'tietoliikenne': 5,
    'tori': 6,
    'urheilu': 7,
    'viestinta': 8
}


def create_response(content, status_code):
    return Response(json.dumps(content), status_code, mimetype='application/json')


def predict_wsgi(environ, start_response):
    global model, tokenizer

    request = Request(environ)
    get_text = request.args.get('text')
    if get_text is None:
        result = {'error': 'No text given in the request.'}
        response = create_response(result, 400)
        return response(environ, start_response)

    texts = [get_text]

    # Load model as global object so it stays in the memory making responses fast.
    if model is None:
        # Try to find HDF5 files on the current directory to load as the model.
        local_hdf5_files = glob.glob('*.h5')
        if not local_hdf5_files:
            result = {'error': 'Could not find predictive model to load, contact support.'}
            response = create_response(result, 400)
            return response(environ, start_response)
        model_path = os.path.join(os.getcwd(), local_hdf5_files[0])
        model = load_model(model_path)

    if tokenizer is None:
        with gzip.open('tokenizer_sfnet.json.gz', 'rt', encoding='utf-8') as f:
            tokenizer = text.text.tokenizer_from_json(f.read())

    print(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    print(sequences)
    data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Give prediction
    predictions = model.predict(data)
    prediction = predictions[0]

    # Report results.
    result = {g: float(prediction[i]) for g, i in groups.items()}
    response = create_response(result, 200)
    return response(environ, start_response)


predict_wsgi = DebuggedApplication(predict_wsgi)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 8000, predict_wsgi)
