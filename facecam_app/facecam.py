import cv2
import os
from deepface import DeepFace
from facecam_app.constants import FACE_FILTER_PATH, KNOWN_FACE_PATH
import hashlib
import asyncio

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
    
    async def cache_image(self, img_path):
        """Функция для вычисления хэша изображения для кэширования (асинхронно)"""
        loop = asyncio.get_event_loop()
        file_hash = await loop.run_in_executor(None, self._get_file_hash, img_path)
        return file_hash

    def _get_file_hash(self, img_path):
        """Вспомогательная функция для вычисления хэша файла (не асинхронная)."""
        with open(img_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    async def compare_single_face(self, img_face, known_face_path, processed_faces):
        """Сравнение одного лица с одним известным лицом (асинхронно)."""
        file_hash = await self.cache_image(known_face_path)

        # Если это изображение уже обработано, пропускаем
        if file_hash in processed_faces:
            label = processed_faces[file_hash]
            color = (0, 255, 0) if label == "СВОЙ" else (0, 0, 255)
        else:
            try:
                result = DeepFace.verify(img_face, known_face_path, detector_backend='opencv')
                if result['verified']:
                    label = "СВОЙ"
                    color = (0, 255, 0)  # Зеленый
                else:
                    label = "ЧУЖОЙ"
                    color = (0, 0, 255)  # Красный

                # Кэшируем результат для текущего изображения
                processed_faces[file_hash] = label
            except Exception as e:
                label = "Ошибка сравнения"
                color = (0, 255, 255)  # Желтый
                print(f"Ошибка DeepFace: {e}")
        
        return label, color, file_hash

    async def compare_faces(self):
        """Сравнение лица в реальном времени с изображениями в папке с известными лицами (асинхронно)."""
        if not os.path.exists(KNOWN_FACE_PATH):
            print(f"Ошибка: Папка {KNOWN_FACE_PATH} не найдена.")
            return

        # Кэшируем результаты для предотвращения повторной обработки тех же изображений
        processed_faces = {}

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: Камера не подключена.")
            return
        
        img_face = "facecam_app/photo/current_face.jpg"
        frame_counter = 0
        skip_frames = 10  # Пропускать 10 кадров

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

                # Проходим по всем изображениям в папке с известными лицами асинхронно
                tasks = []
                for filename in os.listdir(KNOWN_FACE_PATH):
                    known_face_path = os.path.join(KNOWN_FACE_PATH, filename)
                    if os.path.isfile(known_face_path):
                        tasks.append(self.compare_single_face(img_face, known_face_path, processed_faces))

                results = await asyncio.gather(*tasks)

                for label, color, file_hash in results:
                    # Подсвечиваем лицо рамкой
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

                    # Выводим результат в консоль
                    print(f"Результат сравнения с {filename}: {label}")

            cv2.imshow("Face Verification", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
