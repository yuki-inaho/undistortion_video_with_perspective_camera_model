import cv2
import numpy as np
from ruamel import yaml as yaml_ruamel


def load_yaml(file_path):
    yaml = yaml_ruamel.YAML()
    with open(file_path) as stream:
        yaml_dict = yaml.load(stream)
    return yaml_dict


class Undistortion:
    def __init__(self, config_file_path):
        camera_param_dict = load_yaml(config_file_path)

        self._image_size = (camera_param_dict["image_width"], camera_param_dict["image_height"])
        self._K = np.asarray(camera_param_dict["camera_matrix"]["data"], dtype=float).reshape(3, 3)
        self._D = np.asarray(camera_param_dict["distortion_coefficients"]["data"], dtype=float)
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(self._K, self._D, None, self._K, self._image_size, cv2.CV_16SC2)

    def correction(self, image):
        image_undistorted = cv2.remap(image, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return image_undistorted

    @property
    def image_size(self):
        return self._image_size

    @property
    def K(self):
        return self._K

    @property
    def D(self):
        return self._D