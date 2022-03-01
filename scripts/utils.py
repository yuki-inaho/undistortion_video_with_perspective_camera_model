import shutil
import numpy as np
from pathlib import PosixPath


def make_output_dir(output_image_dir_path: PosixPath, clean=False):
    if output_image_dir_path.exists() and clean:
        shutil.rmtree(str(output_image_dir_path))
    output_image_dir_path.mkdir()


def get_image_path_list(input_rgb_dir_pathlib):
    return np.sort([str(path) for path in input_rgb_dir_pathlib.glob("*") if path.suffix in [".jpg", ".png"]])
