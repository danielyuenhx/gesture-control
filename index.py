import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

vid = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
  ret, frame = vid.read()
  if not ret:
    break

  image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  results = hands.process(image_rgb)

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
      thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

      if index_finger_y < thumb_y:
        hand_genture = 'pointing up'
      elif index_finger_y > thumb_y:
        hand_genture = 'pointing down'
      else:
        hand_genture = 'other'

      if hand_genture == 'pointing up':
        pyautogui.press('volumeup')
      elif hand_genture == 'pointing down':
        pyautogui.press('volumedown')

  cv2.imshow('frame capture', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

vid.release()
cv2.destroyAllWindows()