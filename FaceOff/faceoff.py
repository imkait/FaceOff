import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
from PIL import Image


# 計算FPS的參數
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

# cv2轉pil格式
def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# pil轉cv2格式
def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def run(model: str, min_detection_confidence: float,min_suppression_threshold: float, camera_id: int, width: int,height: int) -> None:
    # 臉譜檔案編號
    i=1

    # 攝影機擷取畫面
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # FPS文字的參數
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(result: vision.FaceDetectorResult, unused_output_image: mp.Image,timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # 計算FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # 初始化臉部偵測模型
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceDetectorOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.LIVE_STREAM,
                                        min_detection_confidence=min_detection_confidence,
                                        min_suppression_threshold=min_suppression_threshold,
                                        result_callback=save_result)
    detector = vision.FaceDetector.create_from_options(options)


    # 當鏡頭開啟時
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('錯誤: 無法讀取webcam. 請確認你的webcam設定。')

        # 翻轉畫面
        image = cv2.flip(image, 1)

        # 轉換影像依據模型需要，從BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 使用模型對影像檢測
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # 顯示 FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,font_size, text_color, font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:
            #如果有偵測到人臉
            if len(DETECTION_RESULT.detections)!=0:
                # 取第一個偵測到的人臉
                detection=DETECTION_RESULT.detections[0]
                # 取得框
                bbox = detection.bounding_box
                # 取得索引值，0為人臉
                idx = detection.categories[0].index
                if idx==0:                    
                    x,y,w,h=bbox.origin_x,bbox.origin_y,bbox.width,bbox.height
                    x=x-int(w/2)
                    y=y-int(h/2)-20
                    # 讀取臉譜
                    face = Image.open(f'./face/{i%6+1}.png')
                    # 調整臉譜寬高
                    face=face.resize((w*2,h*2))
                    # 將目前影像畫面轉成pil格式
                    pil_img = cv2_to_pil(current_frame)
                    # 貼上臉譜
                    pil_img.paste( face , (x,y), face )
                    # 重新將畫面轉回cv2格式
                    current_frame=pil_to_cv2(pil_img)
            else:
                # 臉譜編號+1
                i+=1

        cv2.imshow('FaceOff', current_frame)

        # ESC鍵離開程式
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run('detector.tflite', 0.5, 0.5,int(0), 1280, 720)
    
    #參數說明:
    # 模型檔名: 'face_detector.tflite'
    # 最低信心分數: 0.5 
    # 重疊的非最大抑制門檻:0.5 (數字越小越會出現重疊，越大較不會出現重疊，但可能漏掉較小目標) 
    # 相機編號: 0
    # 畫面寬度: 1280
    # 畫面高度: 720