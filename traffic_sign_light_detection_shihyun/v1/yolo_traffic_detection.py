import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names") as f:
    classes = [line.strip() for line in f]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("test.mp4")
while cap.isOpened():
    ret_, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 3)

    cv2.imshow("result", cv2.resize(frame, (800, 600)))
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
