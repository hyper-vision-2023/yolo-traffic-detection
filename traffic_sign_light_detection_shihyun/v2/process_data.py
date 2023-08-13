import itertools
import json
from pathlib import Path
from tqdm import tqdm


def main():
    INPUT_DIRS = [
        "./unprocessed_datasets_labels/traffic_lights_signs/labels/train_1280_720_daylight_1",
        "./unprocessed_datasets_labels/traffic_lights_signs/labels/validation_1280_720_daylight_1",
    ]
    files = list(
        itertools.chain.from_iterable(
            Path(dir).glob("*.json") for dir in INPUT_DIRS
        )
    )

    for file in tqdm(files):
        with open(file) as f:
            annotations = json.load(f)["annotation"]

        labels = create_labels(annotations)
        if labels:
            path_parts = list(file.parts)
            path_parts[0] = "datasets"
            output_file = Path(*path_parts).with_suffix(".txt")

            with open(output_file, "w") as f:
                for label in labels:
                    f.write(label)
                    f.write("\n")


def create_labels(annotations):
    labels = []

    for annotation in annotations:
        if annotation["class"] == "traffic_sign":
            if annotation["type"] == "warning":
                add_label(labels, 0, annotation)
            elif annotation["type"] == "restriction":
                add_label(labels, 1, annotation)
            elif annotation["type"] == "instruction":
                add_label(labels, 2, annotation)
        elif annotation["class"] == "traffic_light":
            if annotation["type"] == "car":
                traffic_light_color = annotation["attribute"][0]

                if traffic_light_color["red"] == "on":
                    add_label(labels, 3, annotation)
                elif traffic_light_color["yellow"] == "on":
                    add_label(labels, 4, annotation)
                elif traffic_light_color["green"] == "on":
                    add_label(labels, 5, annotation)
                elif traffic_light_color["left_arrow"] == "on":
                    add_label(labels, 6, annotation)
            elif annotation["type"] == "pedestrian":
                traffic_light_color = annotation["attribute"][0]

                if traffic_light_color["red"] == "on":
                    add_label(labels, 7, annotation)
                elif traffic_light_color["green"] == "on":
                    add_label(labels, 8, annotation)

    return labels


def add_label(labels, class_id, annotation):
    x1, y1, x2, y2 = annotation["box"]

    top_left_x = min(x1, x2)
    top_left_y = min(y1, y2)
    bottom_right_x = max(x1, x2)
    bottom_right_y = max(y1, y2)

    # normalize coordinates to [0, 1)
    top_left_x /= 1280
    top_left_y /= 720
    bottom_right_x /= 1280
    bottom_right_y /= 720

    center_x = (top_left_x + bottom_right_x) / 2
    center_y = (top_left_y + bottom_right_y) / 2
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    labels.append(f"{class_id} {center_x} {center_y} {width} {height}")


if __name__ == "__main__":
    main()
