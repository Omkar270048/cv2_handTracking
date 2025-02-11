import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open Webcam
cap = cv2.VideoCapture(0)

# Get Screen Size
screen_width, screen_height = pyautogui.size()

# Smoothing parameters
smoothed_x, smoothed_y = 0, 0
alpha = 0.2  # Smoothing factor (higher = smoother but more lag)

# Control Click Timing
last_click_time = 0
click_delay = 0.5  # 500ms delay between clicks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)
    h, w, _ = frame.shape  # Get frame dimensions

    # Define Interaction Rectangle
    x1, y1 = int(w * 0.6), int(h * 0.6)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Scaling Factors (Convert rectangle space to full screen)
    scale_x = screen_width / (x2 - x1)
    scale_y = screen_height / (y2 - y1)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pointer (Index Finger Tip)
            pointer_tip = hand_landmarks.landmark[8]
            pointer_x, pointer_y = int(pointer_tip.x * w), int(pointer_tip.y * h)

            # Pointer Base (Landmark 5)
            pointer_base = hand_landmarks.landmark[5]
            pointer_base_x, pointer_base_y = int(pointer_base.x * w), int(pointer_base.y * h)

            # Thumb Tip (Landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Click Detection with Delay
            pos = pointer_base_x - thumb_x
            current_time = pyautogui.time.time()
            if pos < 5 and (current_time - last_click_time > click_delay):
                pyautogui.click()
                last_click_time = current_time
                print("Left Click")
            elif pos > 50 and (current_time - last_click_time > click_delay):
                pyautogui.doubleClick()
                last_click_time = current_time
                print("Double Click")

            # Ensure the pointer is inside the rectangle
            if x1 <= pointer_x <= x2 and y1 <= pointer_y <= y2:
                norm_x = (pointer_x - x1) / (x2 - x1)
                norm_y = (pointer_y - y1) / (y2 - y1)

                # Map normalized values to full screen
                target_x = int(norm_x * screen_width)
                target_y = int(norm_y * screen_height)

                # Smooth Movement
                smoothed_x = alpha * target_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * target_y + (1 - alpha) * smoothed_y

                # Move Mouse Smoothly
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            # Draw Pointer on Frame
            cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open Webcam
cap = cv2.VideoCapture(0)

# Get Screen Size
screen_width, screen_height = pyautogui.size()

# Smoothing parameters
smoothed_x, smoothed_y = 0, 0
alpha = 0.2  # Smoothing factor (higher = smoother but more lag)

# Control Click Timing
last_click_time = 0
click_delay = 0.5  # 500ms delay between clicks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)
    h, w, _ = frame.shape  # Get frame dimensions

    # Define Interaction Rectangle
    x1, y1 = int(w * 0.6), int(h * 0.6)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Scaling Factors (Convert rectangle space to full screen)
    scale_x = screen_width / (x2 - x1)
    scale_y = screen_height / (y2 - y1)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pointer (Index Finger Tip)
            pointer_tip = hand_landmarks.landmark[8]
            pointer_x, pointer_y = int(pointer_tip.x * w), int(pointer_tip.y * h)

            # Pointer Base (Landmark 5)
            pointer_base = hand_landmarks.landmark[5]
            pointer_base_x, pointer_base_y = int(pointer_base.x * w), int(pointer_base.y * h)

            # Thumb Tip (Landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Click Detection with Delay
            pos = pointer_base_x - thumb_x
            current_time = pyautogui.time.time()
            if pos < 5 and (current_time - last_click_time > click_delay):
                pyautogui.click()
                last_click_time = current_time
                print("Left Click")
            elif pos > 50 and (current_time - last_click_time > click_delay):
                pyautogui.doubleClick()
                last_click_time = current_time
                print("Double Click")

            # Ensure the pointer is inside the rectangle
            if x1 <= pointer_x <= x2 and y1 <= pointer_y <= y2:
                norm_x = (pointer_x - x1) / (x2 - x1)
                norm_y = (pointer_y - y1) / (y2 - y1)

                # Map normalized values to full screen
                target_x = int(norm_x * screen_width)
                target_y = int(norm_y * screen_height)

                # Smooth Movement
                smoothed_x = alpha * target_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * target_y + (1 - alpha) * smoothed_y

                # Move Mouse Smoothly
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            # Draw Pointer on Frame
            cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open Webcam
cap = cv2.VideoCapture(0)

# Get Screen Size
screen_width, screen_height = pyautogui.size()

# Smoothing parameters
smoothed_x, smoothed_y = 0, 0
alpha = 0.2  # Smoothing factor (higher = smoother but more lag)

# Control Click Timing
last_click_time = 0
click_delay = 0.5  # 500ms delay between clicks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)
    h, w, _ = frame.shape  # Get frame dimensions

    # Define Interaction Rectangle
    x1, y1 = int(w * 0.6), int(h * 0.6)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Scaling Factors (Convert rectangle space to full screen)
    scale_x = screen_width / (x2 - x1)
    scale_y = screen_height / (y2 - y1)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pointer (Index Finger Tip)
            pointer_tip = hand_landmarks.landmark[8]
            pointer_x, pointer_y = int(pointer_tip.x * w), int(pointer_tip.y * h)

            # Pointer Base (Landmark 5)
            pointer_base = hand_landmarks.landmark[5]
            pointer_base_x, pointer_base_y = int(pointer_base.x * w), int(pointer_base.y * h)

            # Thumb Tip (Landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Click Detection with Delay
            pos = pointer_base_x - thumb_x
            current_time = pyautogui.time.time()
            if pos < 5 and (current_time - last_click_time > click_delay):
                pyautogui.click()
                last_click_time = current_time
                print("Left Click")
            elif pos > 50 and (current_time - last_click_time > click_delay):
                pyautogui.doubleClick()
                last_click_time = current_time
                print("Double Click")

            # Ensure the pointer is inside the rectangle
            if x1 <= pointer_x <= x2 and y1 <= pointer_y <= y2:
                norm_x = (pointer_x - x1) / (x2 - x1)
                norm_y = (pointer_y - y1) / (y2 - y1)

                # Map normalized values to full screen
                target_x = int(norm_x * screen_width)
                target_y = int(norm_y * screen_height)

                # Smooth Movement
                smoothed_x = alpha * target_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * target_y + (1 - alpha) * smoothed_y

                # Move Mouse Smoothly
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            # Draw Pointer on Frame
            cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open Webcam
cap = cv2.VideoCapture(0)

# Get Screen Size
screen_width, screen_height = pyautogui.size()

# Smoothing parameters
smoothed_x, smoothed_y = 0, 0
alpha = 0.2  # Smoothing factor (higher = smoother but more lag)

# Control Click Timing
last_click_time = 0
click_delay = 0.5  # 500ms delay between clicks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)
    h, w, _ = frame.shape  # Get frame dimensions

    # Define Interaction Rectangle
    x1, y1 = int(w * 0.6), int(h * 0.6)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Scaling Factors (Convert rectangle space to full screen)
    scale_x = screen_width / (x2 - x1)
    scale_y = screen_height / (y2 - y1)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pointer (Index Finger Tip)
            pointer_tip = hand_landmarks.landmark[8]
            pointer_x, pointer_y = int(pointer_tip.x * w), int(pointer_tip.y * h)

            # Pointer Base (Landmark 5)
            pointer_base = hand_landmarks.landmark[5]
            pointer_base_x, pointer_base_y = int(pointer_base.x * w), int(pointer_base.y * h)

            # Thumb Tip (Landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Click Detection with Delay
            pos = pointer_base_x - thumb_x
            current_time = pyautogui.time.time()
            if pos < 5 and (current_time - last_click_time > click_delay):
                pyautogui.click()
                last_click_time = current_time
                print("Left Click")
            elif pos > 50 and (current_time - last_click_time > click_delay):
                pyautogui.doubleClick()
                last_click_time = current_time
                print("Double Click")

            # Ensure the pointer is inside the rectangle
            if x1 <= pointer_x <= x2 and y1 <= pointer_y <= y2:
                norm_x = (pointer_x - x1) / (x2 - x1)
                norm_y = (pointer_y - y1) / (y2 - y1)

                # Map normalized values to full screen
                target_x = int(norm_x * screen_width)
                target_y = int(norm_y * screen_height)

                # Smooth Movement
                smoothed_x = alpha * target_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * target_y + (1 - alpha) * smoothed_y

                # Move Mouse Smoothly
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            # Draw Pointer on Frame
            cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open Webcam
cap = cv2.VideoCapture(0)

# Get Screen Size
screen_width, screen_height = pyautogui.size()

# Smoothing parameters
smoothed_x, smoothed_y = 0, 0
alpha = 0.2  # Smoothing factor (higher = smoother but more lag)

# Control Click Timing
last_click_time = 0
click_delay = 0.5  # 500ms delay between clicks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)
    h, w, _ = frame.shape  # Get frame dimensions

    # Define Interaction Rectangle
    x1, y1 = int(w * 0.6), int(h * 0.6)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Scaling Factors (Convert rectangle space to full screen)
    scale_x = screen_width / (x2 - x1)
    scale_y = screen_height / (y2 - y1)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pointer (Index Finger Tip)
            pointer_tip = hand_landmarks.landmark[8]
            pointer_x, pointer_y = int(pointer_tip.x * w), int(pointer_tip.y * h)

            # Pointer Base (Landmark 5)
            pointer_base = hand_landmarks.landmark[5]
            pointer_base_x, pointer_base_y = int(pointer_base.x * w), int(pointer_base.y * h)

            # Thumb Tip (Landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Click Detection with Delay
            pos = pointer_base_x - thumb_x
            current_time = pyautogui.time.time()
            if pos < 5 and (current_time - last_click_time > click_delay):
                pyautogui.click()
                last_click_time = current_time
                print("Left Click")
            elif pos > 50 and (current_time - last_click_time > click_delay):
                pyautogui.doubleClick()
                last_click_time = current_time
                print("Double Click")

            # Ensure the pointer is inside the rectangle
            if x1 <= pointer_x <= x2 and y1 <= pointer_y <= y2:
                norm_x = (pointer_x - x1) / (x2 - x1)
                norm_y = (pointer_y - y1) / (y2 - y1)

                # Map normalized values to full screen
                target_x = int(norm_x * screen_width)
                target_y = int(norm_y * screen_height)

                # Smooth Movement
                smoothed_x = alpha * target_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * target_y + (1 - alpha) * smoothed_y

                # Move Mouse Smoothly
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            # Draw Pointer on Frame
            cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
