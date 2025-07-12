import os
import cv2

DATASET_PATH = 'dataset'

FOLDERS = ['train', 'val']

IMG_EXTS = ['.jpg', '.jpeg', '.png']

def review_images():
    for folder in FOLDERS:
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        print(f'\nReviewing folder: {folder_path}')
        for class_name in sorted(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_path):
                continue
            print(f'  Reviewing class: {class_name}')
            images = [f for f in os.listdir(class_path) if os.path.splitext(f)[1].lower() in IMG_EXTS]
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f'    [!] Could not open {img_path}, skipping.')
                    continue
                cv2.imshow(f'Reviewing: {class_name}', img)
                print(f'    {img_name}: [d=delete, k=keep, q=quit class]')
                while True:
                    key = cv2.waitKey(0)
                    if key == ord('d'):
                        os.remove(img_path)
                        print(f'      Deleted {img_name}')
                        break
                    elif key == ord('k'):
                        print(f'      Kept {img_name}')
                        break
                    elif key == ord('q'):
                        print(f'      Quitting class {class_name}')
                        cv2.destroyAllWindows()
                        return
                cv2.destroyAllWindows()
    print('Review complete!')

if __name__ == '__main__':
    review_images() 