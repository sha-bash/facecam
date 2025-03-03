import cv2
from facecam_app.constants import FACE_FILTER_PATH

def face_capture():
    clf = cv2.CascadeClassifier(FACE_FILTER_PATH)
    if clf.empty():
        print("Ошибка: Не удалось загрузить каскадный классификатор.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Камера не подключена.")
        return
    
    face_detected = False  # Флаг наличия лица в кадре

    while True:
        ret, img = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить изображение с камеры.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(gray, 
                                     scaleFactor=1.1, 
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(faces) > 0 and not face_detected:
            print("Лицо появилось!")
            face_detected = True 
        elif len(faces) == 0 and face_detected:
            print("Лицо исчезло!")
            face_detected = False  

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('faces', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
