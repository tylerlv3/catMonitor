from ultralytics import YOLO


model = YOLO('yolov8n.pt')
CAT_CLASS_ID = model.names.inverse['cat']


def detect(frame):
    results = model(frame, verbose=False)

    cat_boxes = []

    for result in results:
        for box in result.boxes:
            if box.cls == CAT_CLASS_ID:
                x, y, w, h = box.xywh[0]
                x1 = int(x-w /2)
                y1 = int(y-h /2)

                cat_boxes.append((x1, y1, w, h))

    return cat_boxes





