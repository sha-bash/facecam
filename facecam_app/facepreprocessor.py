import cv2
import os
from constants import FACE_FILTER_PATH

class FacePreprocessor:
    def __init__(self):
        self.clf = cv2.CascadeClassifier(FACE_FILTER_PATH)
        if self.clf.empty():
            raise ValueError("Ошибка: Не удалось загрузить каскадный классификатор.")
        
        self.resize_dim = (160, 160)  # Определяем размер для оптимизации

    def detect_face(self, img):
        """Метод для обнаружения лица на изображении."""
        gray = img
        # Если изображение цветное (3 канала), преобразуем его в оттенки серого
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.clf.detectMultiScale(gray, 
                                          scaleFactor=1.1, 
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        return faces

    def preprocess_and_save(self, img_path, output_folder):
        """Обрабатывает изображение, изменяет размер и сохраняет вырезанное лицо в формате .jpg с качеством 85%."""
        img = cv2.imread(img_path)

        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение {img_path}")
            return

        # Преобразование изображения в серые оттенки для ускорения обработки
        faces = self.detect_face(img)

        if len(faces) > 0:
            # Берем первое найденное лицо
            (x, y, w, h) = faces[0]
            face_crop = img[y:y+h, x:x+w]

            # Изменение размера для стандартизации изображений
            face_resized = cv2.resize(face_crop, self.resize_dim)

            # Генерация имени для сохранения
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, filename)

            # Сохраняем изображение в формате .jpg с качеством 85%
            cv2.imwrite(output_path, face_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            print(f"Сохранено лицо из {img_path} как {output_path}")
        else:
            print(f"Лицо не найдено на изображении {img_path}")

    def process_faces_in_folder(self, folder_path, output_folder):
        """Обрабатывает все изображения в папке и сохраняет вырезанные и измененные лица в формате .jpg с качеством 85%."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Проходим по всем изображениям в папке
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)

            if os.path.isfile(img_path):
                self.preprocess_and_save(img_path, output_folder)

if __name__ == "__main__":
    preprocessor = FacePreprocessor()
    
    folder_with_images = 'facecam_app/photo/faces'  
    processed_folder = 'facecam_app/photo/known_faces'  

    preprocessor.process_faces_in_folder(folder_with_images, processed_folder)
