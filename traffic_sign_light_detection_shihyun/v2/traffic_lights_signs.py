import cv2
from enum import Enum
import numpy as np
from threading import Thread
import time
import timeit
import torch


def main():
    capture_thread = Thread(target=video_capture)
    capture_thread.start()

    do_detection()
    capture_thread.join()


State = Enum("State", ["NotStarted", "Processing", "Stopped"])

image = None
state = State.NotStarted


def video_capture():
    global image
    global state

    cap = cv2.VideoCapture("test.mp4")
    max_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    prev = 0

    while state != State.Stopped and cap.isOpened():
        elapsed = timeit.default_timer() - prev
        difference = 1 / max_frame_rate - elapsed

        if difference > 0:
            time.sleep(difference)
            continue

        prev = timeit.default_timer()

        _, image = cap.read()

    cap.release()
    state = State.Stopped


def do_detection():
    global image
    global state

    model = torch.hub.load(
        "yolov5",
        "custom",
        path="runs/train/exp/weights/best.pt",
        source="local",
        device="mps",
    )

    max_frame_rate = 60
    prev = 0

    state = State.Processing
    while state != State.Stopped:
        elapsed = timeit.default_timer() - prev
        difference = 1 / max_frame_rate - elapsed

        if difference > 0:
            time.sleep(difference)
            continue

        prev = timeit.default_timer()

        if image is None:
            continue

        results = model(image)
        # print(results)
        dataframe = results.pandas().xyxy[0]

        for detection in dataframe.itertuples():
            # e.g. Pandas(Index=0, xmin=824, ymin=148, xmax=855, ymax=183, confidence=0.288, _6=0, name='traffic_sign'))
            print(detection)

        try:
            image = results.render()[0]
        except KeyError:
            pass

        cv2.imshow("result", image)
        if cv2.waitKey(1) == ord("q"):
            state = State.Stopped

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
