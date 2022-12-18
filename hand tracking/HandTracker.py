import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

webbrowser.open('https://freepacman.org/')

time.sleep(1)

pyautogui.moveTo(545, 360)
pyautogui.click()


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                if id ==8:
                    POINTER_TIP = lm
                if id ==5:
                    POINTER_BASE = lm
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        xDifference = POINTER_TIP.x - POINTER_BASE.x
        yDifference = POINTER_TIP.y - POINTER_BASE.y
        if abs(xDifference) > abs(yDifference) :
            if xDifference > 0:
                pyautogui.press('left')
            else: 
                pyautogui.press('right')
        else:
            if yDifference > 0:
                pyautogui.press('down')
            else: 
                pyautogui.press('up')

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()