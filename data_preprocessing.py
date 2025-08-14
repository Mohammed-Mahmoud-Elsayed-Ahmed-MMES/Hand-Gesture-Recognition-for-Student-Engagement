import cv2
import mediapipe as mp
import os
import config
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

def crop_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
    )

    # Ensure output folder exists
    os.makedirs(config.OUTPUT_CROPPED_FOLDER, exist_ok=True)

    # Process each class folder (e.g., dislike, like)
    for class_name in os.listdir(config.INPUT_FOLDER):
        class_path = os.path.join(config.INPUT_FOLDER, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(config.OUTPUT_CROPPED_FOLDER, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        # Process each image
        for idx, img_name in enumerate(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        h, w, _ = image.shape
                        x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w - 20))
                        x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w + 20))
                        y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h - 20))
                        y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h + 20))

                        hand_crop = image[y_min:y_max, x_min:x_max]
                        if hand_crop.size == 0:
                            print(f"Empty crop for image: {img_path}, hand {hand_idx}")
                            continue

                        output_path = os.path.join(output_class_path, f"hand_{idx}_{hand_idx}.jpg")
                        cv2.imwrite(output_path, hand_crop)
                        print(f"Saved hand image: {output_path}")
                else:
                    print(f"No hands detected in image: {img_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")

    hands.close()
    print("Hand cropping completed.")

def split_dataset():
    # Create split directories
    for split in ['train', 'val', 'test']:
        for class_name in config.CLASS_NAMES.values():
            os.makedirs(os.path.join(config.OUTPUT_SPLIT_FOLDER, split, class_name), exist_ok=True)

    # Collect all image paths and labels
    image_paths = []
    labels = []
    for class_name in os.listdir(config.OUTPUT_CROPPED_FOLDER):
        class_path = os.path.join(config.OUTPUT_CROPPED_FOLDER, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(list(config.CLASS_NAMES.values()).index(class_name))

    # Split dataset (stratified to preserve class distribution)
    try:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=(config.VAL_RATIO + config.TEST_RATIO),
            stratify=labels, random_state=42
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
            stratify=temp_labels, random_state=42
        )
    except Exception as e:
        print(f"Error during dataset splitting: {str(e)}")
        return

    # Copy images to respective split folders
    for paths, split_labels, split in [(train_paths, train_labels, 'train'), (val_paths, val_labels, 'val'), (test_paths, test_labels, 'test')]:
        for img_path, label in zip(paths, split_labels):
            class_name = list(config.CLASS_NAMES.values())[label]
            dest_path = os.path.join(config.OUTPUT_SPLIT_FOLDER, split, class_name, os.path.basename(img_path))
            try:
                shutil.copy(img_path, dest_path)
                print(f"Copied {os.path.basename(img_path)} to {split}/{class_name}")
            except Exception as e:
                print(f"Failed to copy {os.path.basename(img_path)} to {split}/{class_name}: {e}")

    # Verify files exist after copying
    for split in ['train', 'val', 'test']:
        for class_name in config.CLASS_NAMES.values():
            folder_path = os.path.join(config.OUTPUT_SPLIT_FOLDER, split, class_name)
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"{split}/{class_name} contains {len(files)} images")

if __name__ == "__main__":
    crop_hands()
    split_dataset()
    print("Dataset splitting completed.")