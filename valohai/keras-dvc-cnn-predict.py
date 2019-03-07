#!/usr/bin/env python3

import argparse
import json
import os
import requests
import tempfile

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

tmp_dir = '/tmp'


def download_url(url, target_dir):
    suffix = os.path.splitext(url)[1]
    fp = tempfile.NamedTemporaryFile(dir=target_dir, suffix=suffix, delete=False)
    r = requests.get(url, allow_redirects=True)
    fp.write(r.content)
    fp.close()
    return fp.name


def main(args):
    model = load_model(args.model)
    # print(model.summary())
    print('Loaded model [{}] with {} layers.\n'.format(args.model, len(model.layers)))

    td = tempfile.TemporaryDirectory(dir=tmp_dir)
    image_dir = td.name
    target_dir = os.path.join(image_dir, 'a')
    os.makedirs(target_dir)

    print('Downloading images from [{}] to [{}].'.format(args.urls_file, target_dir))
    file_to_url = {}
    with open(args.urls_file, 'r') as fp:
        for url in fp:
            url = url.rstrip()
            fn = os.path.basename(download_url(url, target_dir))
            file_to_url[fn] = url
            # print('*', url)

    print()
    input_image_size = (150, 150)
    noopgen = ImageDataGenerator(rescale=1./255)
    batch_size = 25

    test_generator = noopgen.flow_from_directory(
        image_dir,
        target_size=input_image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    preds = model.predict_generator(test_generator,
                                    steps=len(file_to_url) // batch_size + 1,
                                    use_multiprocessing=False,
                                    workers=4,
                                    verbose=1)

    print()
    filenames = test_generator.filenames
    for i, p in enumerate(preds):
        pn = p[0]
        url = file_to_url[os.path.basename(filenames[i])]
        cls = 'cat' if pn < 0.5 else 'dog'
        print(json.dumps({'url': url, 'value': float(pn), 'class': cls}))

    td.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('urls_file', type=str)
    parser.add_argument('--model', type=str, default='dvc-vgg16-finetune.h5')
    args = parser.parse_args()

    print('Using Keras version:', keras.__version__)
    print()
    main(args)
