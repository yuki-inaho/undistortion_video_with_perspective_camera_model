import cv2
import click
from tqdm import tqdm
from pathlib import Path
from typing import List
from scripts.undistortion import Undistortion
from scripts.utils import get_image_path_list, make_output_dir

SCRIPT_DIR = str(Path().parent)


@click.command()
@click.option("--input-dir-path", "-i", type=str, default=f"{SCRIPT_DIR}/images")
@click.option("--config-yaml-path", "-c", type=str, default=f"{SCRIPT_DIR}/config/cam0.yml")
@click.option("--output-dir-name", "-o", type=str, default="undistorted")
@click.option("--use-fisheye-model", "-f", is_flag=True)
def main(input_dir_path, config_yaml_path, output_dir_name, use_fisheye_model):
    """Set video reader"""
    input_image_path_list: List[str] = get_image_path_list(Path(input_dir_path))
    n_flames = len(input_image_path_list)
    print(f"Number of Frames: {n_flames}")

    """ Set video writer
    """
    output_dir_path = str(Path(input_dir_path).parent.joinpath(output_dir_name))
    make_output_dir(Path(output_dir_path))
    print(f"Output destination: {output_dir_path}")

    """ Set undistortion module
    """
    undistortion_module = Undistortion(config_yaml_path, use_fisheye_model)

    """ Lens distortion correction
    """
    for input_image_path in tqdm(input_image_path_list):
        base_name = Path(input_image_path).name
        image = cv2.imread(input_image_path)
        image_undist = undistortion_module.correction(image)
        output_image_path = str(Path(output_dir_path, base_name))
        cv2.imwrite(output_image_path, image_undist)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
