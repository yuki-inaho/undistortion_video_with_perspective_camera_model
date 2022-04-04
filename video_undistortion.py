import cv2
import click
from tqdm import trange
from pathlib import Path
from scripts.undistortion import Undistortion

SCRIPT_DIR = str(Path().parent)


@click.command()
@click.option("--input-video-path", "-i", type=str, default=f"{SCRIPT_DIR}/movie.mp4")
@click.option("--config-yaml-path", "-c", type=str, default=f"{SCRIPT_DIR}/config/cam0.yml")
@click.option("--output-video-name", "-o", type=str, default="movie_undist.mp4")
@click.option("--frame-rate", "-r", type=int, default=30)
@click.option("--use-fisheye-model", "-f", is_flag=True)
def main(input_video_path, config_yaml_path, output_video_name, frame_rate, use_fisheye_model):
    """Set video reader"""
    reader = cv2.VideoCapture(input_video_path)
    n_flames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of Frames: {n_flames}")

    """ Set video writer
    """
    ret, frame = reader.read()
    size = (frame.shape[1], frame.shape[0])
    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    output_video_path = str(Path(input_video_path).parent.joinpath(output_video_name))
    print(f"Output destination: {output_video_path}")
    writer = cv2.VideoWriter(output_video_path, fmt, frame_rate, size)

    """ Set undistortion module
    """
    undistotion_module = Undistortion(config_yaml_path, use_fisheye_model)

    """ Lens distortion correction
    """
    for _ in trange(1, n_flames - 1):
        ret, frame = reader.read()
        if ret:
            writer.write(undistotion_module.correction(frame))

    reader.release()
    writer.release()


if __name__ == "__main__":
    main()
