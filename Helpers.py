import os

def validate_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
