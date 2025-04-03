import cv2
import time
import pyttsx3
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)

hd = HandDetector(maxHands=1, detectionCon=0.8)

engine = pyttsx3.init()

prev_x = None          
last_greet_time = 0    
greet_delay = 1.5

def detect_sign(hand):
    fingers = hd.fingersUp(hand)  

    gesture_map = {
        (0, 1, 0, 0, 0): "One",
        (0, 1, 1, 0, 0): "Two",
        (0, 1, 1, 1, 0): "Three",
        (1, 0, 0, 0, 1): "Call",
        (1, 1, 1, 1, 1): "Hello",
        (1, 0, 0, 0, 0): "Yes",
        (0, 0, 0, 0, 1): "No",
        (0, 1, 1, 1, 1): "Thank You",
        (1, 0, 0, 1, 1): "Love",
        (1, 1, 0, 0, 0): "Danger",
        (0, 1, 0, 1, 1): "Fire",
        (1, 1, 1, 0, 0): "Water",
        (0, 1, 0, 0, 1): "Rock",
        (1, 1, 1, 1, 0): "High Five",
        (1, 1, 0, 1, 0): "Good Luck",
        (1, 0, 1, 1, 1): "Victory",
        (1, 0, 1, 0, 1): "Magic",
        (0, 0, 1, 1, 1): "Salute",
        (1, 0, 0, 1, 0): "Okay",
        (0, 1, 0, 1, 0): "Stop",
        (1, 1, 0, 1, 1): "Peace",
        (0, 1, 1, 0, 1): "Help",
        (1, 1, 1, 1, 1): "Welcome",
        (1, 1, 0, 1, 1): "Live Long",
        (0, 0, 1, 1, 0): "Fist Bump",
        (1, 1, 1, 0, 1): "Power",
        (1, 0, 1, 1, 0): "Clap",
        (0, 1, 1, 1, 0): "Pray",
        (1, 0, 1, 0, 0): "Punch"
    }

    return gesture_map.get(tuple(fingers), None)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 1)  
    hands, _ = hd.findHands(frame, draw=True, flipType=False)  

    if hands:  
        hand = hands[0]  
        sign = detect_sign(hand)  

        if 'bbox' in hand:
            x, y, w, h = hand['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cx, cy = hand['center']
        if prev_x is not None:
            movement = abs(cx - prev_x)
            if movement > 50 and (time.time() - last_greet_time) > greet_delay:
                sign = "Hello"
        
        if sign and (time.time() - last_greet_time) > greet_delay:
            cv2.putText(frame, sign, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            engine.say(sign)
            engine.runAndWait()
            last_greet_time = time.time()
        
        prev_x = cx  
    else:
        prev_x = None

    cv2.imshow("Webcam Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
