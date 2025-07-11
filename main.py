import screenStream
import classifier
import detector
import utils
import cv2


def main():
    screen = screenStream.ScreenStream()
    while True:
        frame = screen.readFrame()
        if frame is not None:
            cat_box = detector.detect(frame)
            if cat_box is not None:
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