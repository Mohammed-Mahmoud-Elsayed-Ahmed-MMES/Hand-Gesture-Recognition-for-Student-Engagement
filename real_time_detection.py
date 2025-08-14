import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import config

def real_time_detection():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
    )
    mp_drawing = mp.solutions.drawing_utils

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # Load model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config.DROPOUT_RATE),
        nn.Linear(num_ftrs, config.NUM_CLASSES)
    )
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True))
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    model = model.to(config.DEVICE)
    model.eval()

    # Class labels
    class_names = config.CLASS_NAMES

    # Webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)  # Try alternative index
        if not cap.isOpened():
            raise ValueError("No webcam available")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Resize frame
            frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Initialize variables
            label = "No hand detected"
            confidence = 0.0
            bbox = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w - 20))
                    x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w + 20))
                    y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h - 20))
                    y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h + 20))

                    hand_crop = frame[y_min:y_max, x_min:x_max]
                    if hand_crop.size == 0:
                        print(f"Empty crop detected at frame")
                        continue
                    hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                    hand_pil = Image.fromarray(hand_crop_rgb)
                    hand_tensor = preprocess(hand_pil).unsqueeze(0).to(config.DEVICE)

                    with torch.no_grad():
                        output = model(hand_tensor)
                        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                        pred = np.argmax(prob)
                        label = class_names[pred]
                        confidence = prob[pred]

                    bbox = (x_min, y_min, x_max, y_max)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Hand Gesture Recognition', frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    real_time_detection()