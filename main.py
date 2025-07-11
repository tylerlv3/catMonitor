import screenStream
import classifier
import detector
import utils
import cv2
import time


def main():
    screen = screenStream.readFrame()
    last_processed_frame = time.time()
    processing_delay = 2

    while True:
        frame = screen.readFrame()
        if frame is not None:
            current_time = time.time()
            if current_time - last_processed_frame < processing_delay:
                continue
            last_processed_frame = current_time

            cat_box = detector.detect(frame)
            if cat_box:
                for box in cat_box:
                    class_name = classify_cat(frame, box)
                    if class_name is not None:
                        label = f"Cat: {class_name}"
                        utils.draw_info(frame, box, label)

            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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