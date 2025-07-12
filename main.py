import screenStream
import classifier
import detector
import utils
import cv2
import time
import os
import uuid

MANUAL_CLASSIFICATION = False

if MANUAL_CLASSIFICATION:
    from pynput import keyboard
    import threading

DATASET_PATH = "dataset/train"

ACTIVITIES = {"e" : "eating",
              "p" : "playing",
              "s" : "sleeping",
              "i" : "idle"}

if MANUAL_CLASSIFICATION:
    shared_state = {
        'latest_roi': None,
        'running': True
    }
    state_lock = threading.Lock()

def setup_directory():
    for activity in ACTIVITIES.values():
        os.makedirs(os.path.join(DATASET_PATH, activity), exist_ok=True)
    print(f"Dataset directory setup complete, ready to save to {DATASET_PATH}")

if MANUAL_CLASSIFICATION:
    def on_press(key):
        """Callback function for the keyboard listener."""
        try:
            key_char = key.char
            if key_char in ACTIVITIES:
                with state_lock:
                    if shared_state['latest_roi'] is not None:
                        activity = ACTIVITIES[key_char]
                        save_path = os.path.join(DATASET_PATH, activity)
                        filename = f"{activity}_{uuid.uuid4().hex[:8]}.jpg"
                        filepath = os.path.join(save_path, filename)
                        cv2.imwrite(filepath, shared_state['latest_roi'])
                        print(f"SUCCESS: Saved ROI for activity: {activity}")
                    else:
                        print("No cat ROI available to save.")
            elif key_char == 'q':
                print("Quit key pressed. Exiting...")
                with state_lock:
                    shared_state['running'] = False
                return False
        except AttributeError:
            pass

def main():
    setup_directory()

    if MANUAL_CLASSIFICATION:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("Global keyboard listener started. You can now classify images from any window.")

        screen = screenStream.ScreenStream()
        while shared_state['running']:
            frame = screen.readFrame()
            if frame is None:
                break
            cat_box = detector.detect(frame)
            if cat_box:
                box = cat_box[0]
                roi = utils.get_roi(frame, box)
                if roi.size > 0:
                    with state_lock:
                        shared_state['latest_roi'] = roi.copy()
                    label = f"Cat Ready to Save"
                    utils.draw_info(frame, box, label)
            cv2.putText(frame, "Use keys globally: 's', 'e', 'p', 'i'. 'q' to quit.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Cat Monitor - Dataset Collection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        print("Main loop has ended. Cleaning up...")
        if listener.is_alive():
            listener.stop()
        listener.join()
        screen.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")
    else:       
        screen = screenStream.ScreenStream()
        while True:
            frame = screen.readFrame()
            if frame is None:
                break
            cat_box = detector.detect(frame)
            if cat_box:
                for box in cat_box:
                    roi = utils.get_roi(frame, box)
                    if roi.size > 0:
                        class_name = classify_cat(frame, box)
                        label = f"{class_name if class_name else 'Cat Detected'}"
                        utils.draw_info(frame, box, label)
            cv2.putText(frame, "Live: Press 'q' to quit.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Cat Monitor - Live Classification", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        screen.release()
        cv2.destroyAllWindows()

def classify_cat(frame, box):
    x, y, w, h = box
    roi = utils.get_roi(frame, box)
    if roi.size > 0:
        class_name = classifier.classify(roi)
        return class_name
    else:
        return None

if __name__ == "__main__":
    main()