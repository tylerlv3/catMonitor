import screenStream
import classifier
import detector
import utils
import cv2
import time
import os
import uuid

DATASET_PATH = "dataset/train"

ACTIVITIES = {"e" : "eating",
              "p" : "playing",
              "s" : "sleeping"}

def setup_directory():
    for activity in ACTIVITIES.values():
        os.makedirs(os.path.join(DATASET_PATH, activity), exist_ok=True)
    print(f"Dataset directory setup complete, ready to save to {DATASET_PATH}")




def main():
    setup_directory()

    screen = screenStream.ScreenStream()
    last_processed_frame = time.time()
    processing_delay = 2
    latest_cat_rois = []

    while True:
        frame = screen.readFrame()
        if frame is not None:
            current_time = time.time()
            if current_time - last_processed_frame > processing_delay:
                last_processed_frame = current_time
                latest_cat_rois.clear()

                cat_box = detector.detect(frame)
                if cat_box:
                    for box in cat_box:
                        roi = utils.get_roi(frame, box)
                        if roi.size > 0:
                            latest_cat_rois.append(roi)
                        label = f"Cat Detected"
                        utils.draw_info(frame, box, label)

            cv2.putText(frame, "Keys: 's' (sleep), 'e' (eat), 'p' (play). 'q' to quit.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Cat Monitor - Dataset Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key in [ord('s'), ord('e'), ord('p')]:
                if not latest_cat_rois:
                    print("No cat detected")
                    continue
                else:
                    key_char = chr(key).lower()
                    activity = ACTIVITIES[key_char]
                    save_path = os.path.join(DATASET_PATH, activity)

                    filename = f"{activity}_{uuid.uuid4().hex[:8]}.jpg"
                    filepath = os.path.join(save_path, filename)

                    cv2.imwrite(filepath, latest_cat_rois[0])
                print(f"Saved {len(latest_cat_rois)} rois for {activity}")


        else:
            print("No frame")
            break
    screen.release()
    cv2.destroyAllWindows()

def classify_cat(frame, box):
    x, y, w, h = box
    roi = utils.get_roi(frame, box)
    if roi.size > 0:
        class_name = classifier.classify(roi)
        print(f"Current activity: {class_name}")
        return class_name
    else:
        print("No activity currently detected")
        return None


if __name__ == "__main__":
    main()