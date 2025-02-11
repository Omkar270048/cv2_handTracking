import cv2
import os
import mediapipe as mp
import pyautogui
import time

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
            
            h,w, _ = frame.shape # get frame dimension

            # Get thumb tip coordinates (Landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            # convert nirmalized coordinates to pixel dimensions
            thumb_x, thumb_y = int(thumb_tip.x*w), int(thumb_tip.y*h)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0,0,255), -1)


            pointer_tip = hand_landmarks.landmark[8]
            pointer_x, pointer_y = int(pointer_tip.x*w), int(pointer_tip.y*h)
            cv2.circle(frame, (pointer_x, pointer_y), 10, (0,255,0), -1)

            if thumb_x-pointer_x <= 10 and thumb_y-pointer_y <= 10:
                cv2.putText(img=frame, text="Volume Down", org=(20,40), fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 1.0,
                        color = (125, 246, 55),
                        thickness = 1   )
                pyautogui.press("volumedown")
                print("Volume Down")

            # else:
            #     print(f"x: {pointer_x-thumb_x} -- y: {thumb_y-pointer_y}")

            # Volume Up: Medium distance
            elif 10 < (pointer_x-thumb_x) < 50 and 10 < (thumb_y - pointer_y) < 50:
                cv2.putText(img=frame, text="Volume Up", org=(20, 40), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1.0, color=(125, 246, 55), thickness=1)
                pyautogui.press("volumeup")
                print("Volume Up")

    # Display the frame
    cv2.imshow("Hand Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()