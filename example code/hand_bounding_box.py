import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

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
            MIN_HAND_SIZE = 0.1  # Adjust as needed (small hands will be ignored)
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

    # Display the frame
    cv2.imshow("Hand Detection Based on Size", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
