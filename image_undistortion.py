import cv2
import click
from tqdm import trange
from pathlib import Path
from scripts.undistortion import Undistortion

SCRIPT_DIR = str(Path().parent)


@click.command()
@click.option("--input-image-path", "-i", type=str, default=f"{SCRIPT_DIR}/data/image.png")
@click.option("--config-yaml-path", "-c", type=str, default=f"{SCRIPT_DIR}/config/cam0.yml")
@click.option("--use-fisheye-model", "-f", is_flag=True)
def main(input_image_path, config_yaml_path, use_fisheye_model):
    """Set undistortion module"""
    undistotion_module = Undistortion(config_yaml_path, use_fisheye_model)

    """ Lens distortion correction
    """
    frame = cv2.imread(input_image_path)
    frame_undist = undistotion_module.correction(frame)
    input_image_pathlib = Path(input_image_path)
    output_image_path = str(
        Path(input_image_pathlib.parent, input_image_pathlib.stem + "_undist" + input_image_pathlib.suffix)
    )
    cv2.imwrite(output_image_path, frame_undist)
    cv2.waitKey(10)


if __name__ == "__main__":
    main()
