from ultralytics import YOLO


model = YOLO('yolov8n.pt')
CAT_CLASS_ID = None
for class_id, class_name in model.names.items():
    if class_name == 'cat':
        CAT_CLASS_ID = class_id
        break


def detect(frame):
    if CAT_CLASS_ID is None:
        raise ValueError("Cat class ID not found")

    results = model(frame, verbose=False)

    cat_boxes = []

    for result in results:
        for box in result.boxes:
            if box.cls == CAT_CLASS_ID:
                tensor_box = box.xywh[0]
                
                x_center = int(tensor_box[0].item())
                y_center = int(tensor_box[1].item())
                w = int(tensor_box[2].item())
                h = int(tensor_box[3].item())

                x1 = int(x_center - w / 2)  
                y1 = int(y_center - h / 2)

                cat_boxes.append((x1, y1, w, h))

    return cat_boxes





