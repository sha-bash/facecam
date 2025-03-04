FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем веса модели DeepFace
RUN mkdir -p /root/.deepface/weights \
    && wget -O /root/.deepface/weights/vgg_face_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5

# Предзагрузка моделей DeepFace
RUN python -c "from deepface import DeepFace; DeepFace.build_model('VGG-Face'); DeepFace.build_model('Facenet'); DeepFace.build_model('OpenFace'); DeepFace.build_model('DeepID'); DeepFace.build_model('Dlib'); DeepFace.build_model('ArcFace')"

CMD ["python", "main.py"]
