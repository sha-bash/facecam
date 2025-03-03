import cv2
import os
from deepface import DeepFace
from facecam_app.constants import FACE_FILTER_PATH, KNOWN_FACE_PATH
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class FaceCam:
    
    def __init__(self):
        self.clf = cv2.CascadeClassifier(FACE_FILTER_PATH)
        if self.clf.empty():
            raise ValueError("Ошибка: Не удалось загрузить каскадный классификатор.")
    
    def detect_face(self, img):
        """Метод для обнаружения лица на изображении."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.clf.detectMultiScale(gray, 
                                          scaleFactor=1.1, 
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        return faces
    
    def compare_faces(self):
        """Сравнение лица в реальном времени с эталонным фото."""
        if not os.path.exists(KNOWN_FACE_PATH):
            print(f"Ошибка: Файл {KNOWN_FACE_PATH} не найден.")
            return
        
        try:
            reference = DeepFace.extract_faces(img_path=KNOWN_FACE_PATH, detector_backend='opencv')[0]['face']
        except Exception as e:
            print(f"Ошибка загрузки эталонного лица: {e}")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: Камера не подключена.")
            return
        
        img_face = "facecam_app/photo/current_face.jpg"

        frame_counter = 0
        skip_frames = 10  

        while True:
            ret, img = cap.read()
            if not ret:
                print("Ошибка: Не удалось получить изображение с камеры.")
                break
            
            if frame_counter % skip_frames == 0:
                faces = self.detect_face(img)

            frame_counter += 1

            if frame_counter % skip_frames != 0:
                continue 
            
            faces = self.detect_face(img)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]  
                face_crop = img[y:y+h, x:x+w] 
                cv2.imwrite(img_face, face_crop)

                try:
                    result = DeepFace.verify(img_face, KNOWN_FACE_PATH, detector_backend='opencv')
                    if result['verified']:
                        label = "СВОЙ"
                        color = (0, 255, 0)  # Зеленый
                    else:
                        label = "ЧУЖОЙ"
                        color = (0, 0, 255)  # Красный
                except Exception as e:
                    label = "Ошибка сравнения"
                    color = (0, 255, 255)  # Желтый
                    print(f"Ошибка DeepFace: {e}")

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                #Выводит метку на изображение
                # pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # draw = ImageDraw.Draw(pil_img)

                # try:
                #     font = ImageFont.truetype("arial.ttf", 32)
                # except IOError:
                #     font = ImageFont.load_default()  
                # draw.text((50, 50), label, font=font, fill=color)
                # img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                #Отображает метку в консоли
                print(f"Результат сравнения: {label}")

            cv2.imshow("Face Verification", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
