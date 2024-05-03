import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt 
import numpy as np
import csv

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #chuyển ảnh từ BGR sang RGB
    image.flags.writeable = False    #không cho sửa ảnh một cách trực tiếp tăng tốc độ xử lý 
    results = model.process(image)    #tạo ra dự đoán 
    image.flags.writeable = True     #cho phép sửa ảnh trong bộ nhớ
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #chuyển ảnh từ RGB sang BGR 
    return image, results


def draw_landmarks(image, results):
    
    # Vẽ tọa độ các điểm tay trái 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Vẽ tọa độ các điểm tay phải 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    #trích xuất giá trị tọa độ các điểm trên tay trái, nếu điểm không xuất hiện trên khung hình thì giá trị là 0 
    lh = list(np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2))
    #trích xuất giá trị tọa độ các điểm trên tay phải, nếu điểm không xuất hiện trên khung hình thì giá trị là 0
    rh = list(np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2))
    return lh, rh 


def collect_keypoints(holistic):
    landmarks_list = []
    data_folder = 'mp_data'
    labels = -1
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            labels += 1
            print(folder_path)
            # Lặp qua các file ảnh trong thư mục con
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                # Kiểm tra nếu là file ảnh
                if img_path.endswith('.jpg') or img_path.endswith('.png'):
                    # Đọc ảnh bằng OpenCV
                    img = cv2.imread(img_path)
                    if img is not None:
                        image, results = mediapipe_detection(img, holistic) 
                        lh_results_list, rh_results_list = extract_keypoints(results)
                        if lh_results_list.count(0) != len(lh_results_list): #nếu như list tọa độ các điểm tay trái trích xuất thành công thì đưa vào data 
                            lh_results_list.insert(0,labels)
                            landmarks_list.append(lh_results_list)
                        if rh_results_list.count(0) != len(rh_results_list):#nếu như list tọa độ các điểm tay phải trích xuất thành công thì đưa vào data
                            rh_results_list.insert(0,labels)
                            landmarks_list.append(rh_results_list)
     

    with open('data1.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(landmarks_list) #đưa data vào file csv


if __name__ == '__main__':
    mp_holistic = mp.solutions.holistic    #sử dụng holistic model 
    mp_drawing = mp.solutions.drawing_utils #vẽ ra màn hình kết quả
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        collect_keypoints(holistic)
        
    