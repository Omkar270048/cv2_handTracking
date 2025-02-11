import cv2
import os
import mediapipe as mp

# Initalize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


   # Convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb tip coordinates (Landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            h,w, _ = frame.shape # get frame dimension

            # convert nirmalized coordinates to pixel dimensions
            thumb_x, thumb_y = int(thumb_tip.x*w), int(thumb_tip.y*h)
            print(f"thumb_tip.x = {thumb_tip.x}  ---  thumb_tip.y = {thumb_tip.y}")
            print(f"thumb_x = {thumb_x}  --- thumb_y = {thumb_y}")

            cv2.circle(frame, (thumb_x, thumb_y), 10, (0,0,255), -1)

    # Display the frame
    cv2.imshow("Hand Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()