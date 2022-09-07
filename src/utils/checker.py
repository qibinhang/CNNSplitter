import os


def check_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
