import cv2
import click
from tqdm import tqdm
from pathlib import Path
from scripts.utils import make_output_dir, get_image_path_list


@click.command()
@click.option("--input-path", "-i", default="./rgb")
@click.option("--output-mp4-path", "-o", default="movie.mp4")
@click.option("--resize-rate", "-r", default=1.0)
@click.option("--frame-rate", "-f", default=10.0)
@click.option("--gray", "-g", is_flag=True)
def main(input_path, output_mp4_path, resize_rate, frame_rate, gray):
    input_image_list = get_image_path_list(Path(input_path))
    n_flames = len(input_image_list)
    print(f"Number of Frame: {n_flames}")

    # @TODO: reduce redundant code lines
    image = cv2.imread(input_image_list[0])
    image_resized = cv2.resize(image, None, fx=resize_rate, fy=resize_rate)
    size = (image_resized.shape[1], image_resized.shape[0])
    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(output_mp4_path, fmt, frame_rate, size)

    for input_image_path in tqdm(input_image_list):
        image = cv2.imread(input_image_path)
        if gray:
            image_gray = cv2.cvtColor(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            image_resized = cv2.resize(image_gray, None, fx=resize_rate, fy=resize_rate)
        else:
            image_resized = cv2.resize(image, None, fx=resize_rate, fy=resize_rate)
        writer.write(image_resized)
        cv2.waitKey(10)
    writer.release()


if __name__ == "__main__":
    main()