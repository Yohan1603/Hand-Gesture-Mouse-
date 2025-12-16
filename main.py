import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Disable pyautogui failsafe for demo
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0

def fingers_up(landmarks):
    # Thumb, index, middle, ring, pinky
    thumb = landmarks[4].x < landmarks[3].x if landmarks[4].x < 0.5 else landmarks[4].x > landmarks[3].x
    fingers = [thumb, landmarks[8].y < landmarks[6].y, landmarks[12].y < landmarks[10].y, 
               landmarks[16].y < landmarks[14].y, landmarks[20].y < landmarks[18].y]
    return fingers

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]
            
            # Smooth cursor movement
            x = int(index_tip.x * screen_w)
            y = int(index_tip.y * screen_h)
            pyautogui.moveTo(x, y, duration=0.1)
            
            fingers = fingers_up(landmarks)
            
            # Left click (thumb + index)
            if fingers[0] and fingers[1]:
                pyautogui.click()
                cv2.putText(frame, "CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            
            # Right click (index + middle)
            elif fingers[1] and fingers[2]:
                pyautogui.rightClick()
                cv2.putText(frame, "RIGHT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            
            # Show finger status
            status = f"Index: {'UP' if fingers[1] else 'DOWN'}"
            cv2.putText(frame, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
    cv2.imshow("Gesture Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
