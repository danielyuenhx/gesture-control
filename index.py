import cv2
import pyautogui
import mediapipe as mp
import time
from enum import Enum
import math

INDEX = 0
MIDDLE = 1
RING = 2
PINKY = 3
THUMB = 4

class Gesture(Enum):
  FIST = 0
  OPEN = 1
  POINT = 2

mp_hands = mp.solutions.hands

def calculate_displacement(point1, point2):
  x_distance = point1.x*100 - point2.x*100
  y_distance = point1.y*100 - point2.y*100
  return math.sqrt(x_distance**2 + y_distance**2)

def calculate_y(point1, point2):
  return point1.y*100 - point2.y*100

def count_fingers(fingers):
  raised_fingers = [0, 0, 0, 0, 0]
  count = 0

  hl = mp_hands.HandLandmark

  # used as a reference distance
  palm_distance = calculate_displacement(fingers[hl.WRIST], fingers[hl.MIDDLE_FINGER_MCP])*2/5
  
  # determine all finger distances
  # index_distance = calculate_displacement(fingers[hl.INDEX_FINGER_MCP], fingers[hl.INDEX_FINGER_TIP])
  # middle_distance = calculate_displacement(fingers[hl.MIDDLE_FINGER_MCP], fingers[hl.MIDDLE_FINGER_TIP])
  # ring_distance = calculate_displacement(fingers[hl.RING_FINGER_MCP], fingers[hl.RING_FINGER_TIP])
  # pinky_distance = calculate_displacement(fingers[hl.PINKY_MCP], fingers[hl.PINKY_TIP])
  # finger_distances = [index_distance, middle_distance, ring_distance, pinky_distance]

  finger_points = [(hl.INDEX_FINGER_MCP, hl.INDEX_FINGER_TIP), (hl.MIDDLE_FINGER_MCP, hl.MIDDLE_FINGER_TIP), (hl.RING_FINGER_MCP, hl.RING_FINGER_TIP), (hl.PINKY_MCP, hl.PINKY_TIP)]

  for i in range(len(finger_points)):
    mcp = fingers[finger_points[i][0]]
    tip = fingers[finger_points[i][1]]
    finger_distance = calculate_y(mcp, tip)
    if finger_distance > palm_distance: 
      raised_fingers[i] = 1
      count += 1
    else:
      raised_fingers[i] = 0

  # x axis for thumb
  thumb_distance = calculate_displacement(fingers[hl.THUMB_TIP], fingers[hl.INDEX_FINGER_MCP])
  print(thumb_distance)
  # if fingers[hl.THUMB_TIP].x > fingers[hl.THUMB_MCP].x:
  if abs(thumb_distance) > 5:
    raised_fingers[THUMB] = 1
    count += 1
  else:
    raised_fingers[THUMB] = 0

  return raised_fingers, count

def start_detection():
  vid = cv2.VideoCapture(0)
  
  hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

  mp_drawing = mp.solutions.drawing_utils
  gesture = None

  countdown = None

  while True:
    ret, frame = vid.read()
    if not ret:
      break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    count = 0

    if results.multi_hand_landmarks:
      # only 1 max_num_hands
      hand_landmark = results.multi_hand_landmarks[0]
      mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

      raised_fingers, count = count_fingers(hand_landmark.landmark)

      # determine the gesture
      if count == 0:
        gesture = Gesture.FIST
      elif count == 5:
        gesture == Gesture.OPEN
      elif raised_fingers[INDEX] and count == 1:
        gesture = Gesture.POINT
      else:
        gesture = None

    cv2.putText(frame, f'Count: {int(count)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3)
    cv2.putText(frame, f'Gesture: {str(gesture)}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3)
    cv2.imshow('frame capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  vid.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  start_detection()
