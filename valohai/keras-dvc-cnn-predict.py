import argparse
import os
import requests

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# image_dir = '/valohai/repository/tmp'
image_dir = os.path.realpath('tmp')


def download_url(url, target_dir):
    output_name = os.path.join(target_dir, os.path.basename(url))
    print('* [{}]'.format(os.path.basename(output_name)))
    r = requests.get(url, allow_redirects=True)
    open(output_name, 'wb').write(r.content)
    return output_name


def main(args):
    model = load_model(args.model)
    # print(model.summary())
    print('Loaded model [{}] with {} layers.\n'.format(args.model, len(model.layers)))

    target_dir = os.path.join(image_dir, 'a')
    os.makedirs(target_dir, exist_ok=True)

    print('Downloading images from [{}] to [{}].'.format(args.urls_file, image_dir))
    image_files = []
    with open(args.urls_file, 'r') as fp:
        for url in fp:
            image_files.append(download_url(url.rstrip(), target_dir))

    print()
    input_image_size = (150, 150)
    noopgen = ImageDataGenerator(rescale=1./255)
    batch_size = 25

    test_generator = noopgen.flow_from_directory(
        os.path.realpath('tmp'),
        target_size=input_image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    preds = model.predict_generator(test_generator,
                                    steps=len(image_files) // batch_size + 1,
                                    use_multiprocessing=False,
                                    workers=4,
                                    verbose=1)

    print()
    filenames = test_generator.filenames
    for i, p in enumerate(preds):
        pn = p[0]
        print(os.path.basename(filenames[i]), pn, 'cat' if pn < 0.5 else 'dog')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('urls_file', type=str)
    parser.add_argument('--model', type=str, default='dvc-vgg16-finetune.h5')
    args = parser.parse_args()

    print('Using Keras version:', keras.__version__)
    print()
    main(args)
