import os
import argparse

import requests

"""
Simple test script that sends target file to local prediction server endpoint
and prints the response status code and content.

If developing the prediction server, remember to start it first

Usage:
    $ python valohai/test_prediction_server_text.py "Hei maailma"
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str)
    args = parser.parse_args()

    params = {'text': args.text}
    response = requests.get('http://localhost:8000', params=params)
    print(f'Result: {response.status_code} => {response.content}')
