import cv2
import pyautogui
import mediapipe as mp
import time
from enum import Enum
import math

pyautogui.PAUSE = 0

INDEX = 0
MIDDLE = 1
RING = 2
PINKY = 3
THUMB = 4

MCP = 0
PIP = 1
DIP = 2
TIP = 3

class Gesture(Enum):
  FIST = 0
  OPEN = 1
  POINT = 2
  SCROLL = 3
  DOWNSCROLL = 4

class Finger:
  def __init__(self, fingers, mcp_index):
    self.points = []
    for i in range(4):
      point = fingers[mcp_index+i]
      self.points.append(Point(point.x, point.y))

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

mp_hands = mp.solutions.hands

def calculate_displacement(point1, point2):
  x_distance = point1.x*100 - point2.x*100
  y_distance = point1.y*100 - point2.y*100
  return math.sqrt(x_distance**2 + y_distance**2)

def calculate_y(point1, point2):
  return point1.y*100 - point2.y*100

def count_fingers(fingers):
  raised_fingers = [0, 0, 0, 0, 0]
  finger_positions = [0, 0, 0, 0, 0]
  count = 0

  hl = mp_hands.HandLandmark

  # used as a reference distance
  palm_distance = calculate_displacement(fingers[hl.WRIST], fingers[hl.MIDDLE_FINGER_MCP])*2/5

  finger_points = [(hl.INDEX_FINGER_MCP, hl.INDEX_FINGER_TIP), (hl.MIDDLE_FINGER_MCP, hl.MIDDLE_FINGER_TIP), (hl.RING_FINGER_MCP, hl.RING_FINGER_TIP), (hl.PINKY_MCP, hl.PINKY_TIP)]

  for i in range(len(finger_points)):
    mcp = fingers[finger_points[i][0]]
    tip = fingers[finger_points[i][1]]

    finger_positions[i] = Finger(fingers, finger_points[i][0])
    finger_distance = calculate_y(mcp, tip)
    if finger_distance > palm_distance and tip.y < mcp.y: 
      raised_fingers[i] = 1
      count += 1
    else:
      raised_fingers[i] = 0

  # x axis for thumb
  thumb_distance = calculate_displacement(fingers[hl.THUMB_TIP], fingers[hl.INDEX_FINGER_MCP])
  finger_positions[THUMB] = Finger(fingers, hl.THUMB_CMC)
  if abs(thumb_distance) > palm_distance*3/4:
    raised_fingers[THUMB] = 1
    count += 1
  else:
    raised_fingers[THUMB] = 0

  return raised_fingers, finger_positions, count


def circle_fingers(frame, fingers):
  frame_height, frame_width, _ = frame.shape

  for finger in fingers:
    cv2.circle(frame, (int(finger.points[TIP].x*frame_width), int(finger.points[TIP].y*frame_height)), 10, (0,255,255))


def start_detection():
  vid = cv2.VideoCapture(0)
  
  screen_width, screen_height = pyautogui.size()
  hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

  mp_drawing = mp.solutions.drawing_utils
  gesture = None

  countdown = None

  scroll_prev_y = 0

  while True:
    ret, frame = vid.read()
    if not ret:
      break
    
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    count = 0

    if results.multi_hand_landmarks:
      # only 1 max_num_hands
      hand_landmark = results.multi_hand_landmarks[0]

      mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

      raised_fingers, finger_positions, count = count_fingers(hand_landmark.landmark)

      # determine the gesture
      if count == 0:
        gesture = Gesture.FIST
      elif count == 5:
        gesture == Gesture.OPEN
      elif raised_fingers[INDEX] and count == 1:
        gesture = Gesture.POINT
      elif raised_fingers[INDEX] and raised_fingers[MIDDLE] and count == 2:
        gesture = Gesture.SCROLL
      elif raised_fingers[INDEX] and raised_fingers[MIDDLE] and raised_fingers[THUMB] and count == 3:
        gesture = Gesture.DOWNSCROLL
      else:
        gesture = None

      x = finger_positions[INDEX].points[TIP].x
      y = finger_positions[INDEX].points[TIP].y
      mouse_x = int(screen_width * x)
      mouse_y = int(screen_height * y)

      if gesture == Gesture.POINT:
        circle_fingers(frame, [finger_positions[INDEX]])
        pyautogui.moveTo(mouse_x, mouse_y)
      elif gesture == Gesture.SCROLL:
        circle_fingers(frame, [finger_positions[INDEX], finger_positions[MIDDLE]])
        speed = (finger_positions[INDEX].points[TIP].y - finger_positions[INDEX].points[PIP].y)*100
        pyautogui.scroll(int(abs(5*speed)))
        # if (scroll_prev_y != 0):
          # pyautogui.scroll(int((y-scroll_prev_y)*1000))
      elif gesture == Gesture.DOWNSCROLL:
        circle_fingers(frame, [finger_positions[INDEX], finger_positions[MIDDLE], finger_positions[THUMB]])
        speed = (finger_positions[INDEX].points[TIP].y - finger_positions[INDEX].points[PIP].y)*100
        pyautogui.scroll(int(-abs(5*speed)))

    cv2.putText(frame, f'Count: {int(count)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3)
    cv2.putText(frame, f'Gesture: {str(gesture)}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3)
    cv2.imshow('frame capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  vid.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  start_detection()
