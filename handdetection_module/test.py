import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Set initial mouse position
mousex, mousey = 200, 200
pyautogui.moveTo(mousex, mousey)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract all x and y coordinates of landmarks
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]

            # Get bounding box size (hand size in image)
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            hand_width = max_x - min_x
            hand_height = max_y - min_y

            # Define hand size thresholds
            MIN_HAND_SIZE = 0.05  # Adjust as needed (small hands will be ignored)
            MAX_HAND_SIZE = 0.7  # Adjust as needed (too large hands will be ignored)

            # Check if the detected hand is within the size range
            if MIN_HAND_SIZE < hand_width < MAX_HAND_SIZE and MIN_HAND_SIZE < hand_height < MAX_HAND_SIZE:
                # Draw hand landmarks if within size range
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Draw a bounding box around the hand
                h, w, _ = frame.shape
                x1, y1 = int(min_x * w), int(min_y * h)
                x2, y2 = int(max_x * w), int(max_y * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

                # Draw lines for reference
                center_x = (x2 + x1) // 2
                cv2.line(frame, (center_x, y1), (center_x, y2), (0, 255, 0), 2)
                cv2.line(frame, (center_x - 10, y1), (center_x - 10, y2), (0, 255, 0), 2)
                cv2.line(frame, (center_x + 10, y1), (center_x + 10, y2), (0, 255, 0), 2)

                # Get thumb tip coordinates (Landmark 4)
                thumb_tip = hand_landmarks.landmark[4]
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Get Pointer tip coordinates (Landmark 8)
                pointer_tip = hand_landmarks.landmark[8]
                pointer_x, pointer_y = int(pointer_tip.x * w), int(pointer_tip.y * h)

                # Get Pointer base coordinates (Landmark 5)
                pointer_base = hand_landmarks.landmark[5]
                pointer_base_x, pointer_base_y = int(pointer_base.x * w), int(pointer_base.y * h)

                # Get middle finger tip coordinates (Landmark 12)
                mf_tip = hand_landmarks.landmark[12]
                mf_x, mf_y = int(mf_tip.x * w), int(mf_tip.y * h)


                dx = center_x - pointer_x
                dy = pointer_base_x - thumb_x

                # Determine hand gestures for cursor movement
                if -10 < dx < 10 and 8 <= dy <= 20:
                    cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 255, 0), -1)  # Green circle (No movement)
                elif -10 > dx and dy < 8:
                    mousex += 10
                    mousey += 10
                elif -10 > dx and dy > 20:
                    mousex += 10
                    mousey -= 10
                elif dx > 10 and dy < 8:
                    mousex -= 10
                    mousey += 10
                elif dx > 10 and dy > 20:
                    mousex -= 10
                    mousey -= 10
                elif -10 > dx:
                    mousex += 10
                elif dx > 10:
                    mousex -= 10
                elif dy < 8:
                    mousey += 10
                elif dy > 20:
                    mousey -= 10

                # Ensure the mouse does not go out of bounds
                mousex = max(1, min(mousex, screen_width - 1))
                mousey = max(1, min(mousey, screen_height - 1))

                # Move the mouse
                pyautogui.moveTo(mousex, mousey)

    # Display the frame
    cv2.imshow("Hand Detection Based on Size", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
