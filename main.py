import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
from face_mesh_point_idxes import beard_point_idxes
from face_mesh_point_idxes import mouse_point_idxes

WIDTH = 640
HEIGHT = 360
FILTER_CNT = 4

def remove_beard(src_image, face_landmarks):
    width = src_image.shape[1]
    height = src_image.shape[0]

    # ぼかす範囲
    beard_boarder_poss = [[face_landmarks.landmark[idx].x * width, face_landmarks.landmark[idx].y * height] for idx in beard_point_idxes]
    # 領域の中心から1.1倍広げる
    beard_boarder_poss = np.array(beard_boarder_poss)
    beard_boarder_center = np.mean(beard_boarder_poss, axis=0)
    beard_boarder_poss = (beard_boarder_poss - beard_boarder_center) * 1.1 + beard_boarder_center

    beard_boarder = np.array([beard_boarder_poss], dtype=np.int32)
    mouse_boarder_poss = [[face_landmarks.landmark[idx].x * width, face_landmarks.landmark[idx].y * height] for idx in mouse_point_idxes]
    mouse_boarder = np.array([mouse_boarder_poss], dtype=np.int32)

    # ぼかし範囲のマスク
    beard_mask = np.zeros(src_image.shape).astype(np.uint8)
    channel_count = image.shape[2]
    white = (255,) * channel_count
    black = (  0,) * channel_count
    cv2.fillPoly(beard_mask, beard_boarder, white)
    cv2.fillPoly(beard_mask, mouse_boarder, black)
    beard_mask_inverse = cv2.bitwise_not(beard_mask)

    # 全体をぼかしたものの一部をソース画像に上書きする
    filtered_image = src_image
    for i in range(FILTER_CNT):
      filtered_image = cv2.bilateralFilter(filtered_image, 8, sigmaColor=100, sigmaSpace=100)
    return cv2.bitwise_and(filtered_image, beard_mask) + cv2.bitwise_and(image, beard_mask_inverse)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Webカメラから入力
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
use_filter = True

# 仮想カメラを作る
v_cam = pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=30)

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if use_filter and results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        image = remove_beard(image, face_landmarks)

    # 処理した画像をウィンドウに表示
    cv2.imshow('MediaPipe Face Mesh', image)

    # 仮想カメラに出力する
    # そのままだと色が変なのでrgbチャネルをbgrに変換
    color_inversed_imgage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    v_cam.send(color_inversed_imgage)

    key = cv2.waitKey(5)
    if key & 0xFF == 27:
      break

    # スペースキーでフィルタの有効を切り替える
    if key & 0xFF == ord(' '):
      use_filter = not use_filter

    # ウィンドウが閉じられたら終了
    try:
      cv2.getWindowProperty('MediaPipe Face Mesh', 0)
    except:
      break
cap.release()