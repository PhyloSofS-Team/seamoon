import os
import json


def load_params(json_file_path):
    with open(json_file_path, "r") as file:
        params = json.load(file)
    # validate_params(params)  # Validate the parameters after loading them #bypassed for now

    return params
