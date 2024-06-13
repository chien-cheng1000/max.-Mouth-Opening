import cv2 as cv
import mediapipe as mp
import numpy as np

#print("mp.__file__", mp.__file__)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles  #繪圖樣式

cap = cv.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
        mp_face_mesh.FaceMesh(refine_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) / 2)  # 記得轉整數!!
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) / 2)
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter('mouth-opening_Ffinal.mp4', cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 20.0,
                         (width, height))  # 這次debug: out要放主循環去寫空的影片，不然每一幀的循環都會執行一次
    while cap.isOpened():
        ret, frame0 = cap.read()
        if not ret:
            break
        # 加強對比度
        contrast = 185
        brightness = 30
        frame = frame0 * (contrast / 127 + 1) - contrast + brightness  # 轉換公式
        # 為了保持色彩區間在0~255的整數範圍中，再用np.clip、np.uint8調整數值
        frame = np.clip(frame, 0, 255)
        frame = np.uint8(frame)  # 確保為0~255的正整數 #資料型態!

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  #轉換成rgb，detection和mesh model都要用rgb偵測
        results_detection = face_detection.process(frame_rgb)
        results_mesh = face_mesh.process(frame_rgb)

        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box  #框取的人臉尺寸
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                if results_mesh.multi_face_landmarks:
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        # 畫輪廓
                        mp_drawing.draw_landmarks(frame0,
                                                  landmark_list=face_landmarks,
                                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                                  landmark_drawing_spec=None,
                                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style
                                                ())

                        #處理嘴角間距離
                        right_mouthCR= face_landmarks.landmark[291]
                        left_mouthCR= face_landmarks.landmark[61]
                        # 處理嘴唇邊界位置點
                        upper_lip_center = face_landmarks.landmark[13]  # 13是上嘴唇的底部
                        lower_lip_center = face_landmarks.landmark[14]  # 14是下嘴唇的上端

                        ulcx, ulcy = int(upper_lip_center.x * iw), int(upper_lip_center.y * ih)
                        llcx, llcy = int(lower_lip_center.x * iw), int(lower_lip_center.y * ih)
                        rmcx, rmcy= int(right_mouthCR.x * iw), int(right_mouthCR.y * ih)
                        lmcx, lmcy= int(left_mouthCR.x * iw), int(left_mouthCR.y * ih)


                        vx = (ulcx + llcx) // 2
                        roundV_1 = np.around(((llcy-ulcy) * 13.5 / w), 1)
                        #contrast = 200
                        #brightness = 30

                        if roundV_1>1:

                            v = frame[ulcy:llcy,vx:vx+1]  #2 dimension array

                            output = cv.cvtColor(v, cv.COLOR_BGR2GRAY)
                            v2 = output.flatten()
                            print(v2)
                            filtered_array= output[output<210]   #布林索引篩選
                            filtered= filtered_array.flatten()  #change to list
                            length_filtered= len(filtered)
                            #print('teeth_distance:',length_filtered)

                        else:
                            print('small distance',roundV_1)
                            length_filtered=0

                        cv.circle(frame0, (ulcx, ulcy), 5, (255, 0, 0), -1)
                        cv.circle(frame0, (llcx, llcy), 5, (0, 0, 255), -1)

                        cv.circle(frame0, (rmcx, rmcy), 5, (255, 0, 0), -1)
                        cv.circle(frame0, (lmcx, lmcy), 5, (0, 0, 255), -1)
                        cv.line(frame0,(ulcx, ulcy),(llcx, llcy),(0,0,0),5)

                        #計算上下嘴唇距離
                        # roundV_lips= np.around(((llcy-ulcy)*5/94),1)
                        roundV_teeth= np.around(((length_filtered) * 13.5 / w ),1)  #14 cm for the glass
                        print('teeth_distance:', length_filtered,roundV_teeth)
                        #print(roundV_lips)
                        #print('mouth corner dist: '+str(rmcx-lmcx))
                        cv.putText(frame0,'Teeth_distance: '+str(roundV_teeth) + "cm",(x,y+h+10),cv.FONT_HERSHEY_SIMPLEX,
                                   1,(0,0,0),2)

                # 繪製人臉邊界框
                #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_2 = cv.resize(frame0, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)/2)))               # 改變圖片尺寸
        out.write(frame_2)  # 將讀取的每一幀圖像寫入空的影片
        cv.imshow('Mouth measurement_2', frame0)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()      # 釋放資源
cv.destroyAllWindows()