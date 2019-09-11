import os
import argparse

import requests

"""
Simple test script that sends target file to local prediction server endpoint
and prints the response status code and content.

If developing the prediction server, remember to start it first

Usage:
    $ python valohai/test_prediction_server.py inputs/cat.jpg,inputs/dog.jpg
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    args = parser.parse_args()
    image_path = args.image_path

    # Support for multiple files using , separator
    if ',' in image_path:
        parts = image_path.split(',')
        full_image_paths = [url for url in parts if len(url) > 0]
    else:
        full_image_paths = [os.path.join(os.getcwd(), args.image_path)]

    for fip in full_image_paths:
        if not os.path.isfile(fip):
            print(f'Could not find a file to send at {fip}')
            exit(1)

    for fip in full_image_paths:
        files = {'media': open(fip, 'rb')}
        response = requests.post('http://localhost:8000', files=files)
        print(f'Target: {fip}')
        print(f'Result: {response.status_code} => {response.content}')
