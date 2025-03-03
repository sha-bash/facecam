import asyncio
from facecam_app.facecam import FaceCam

face_cam = FaceCam()

async def main():
    # Инициализируем объект FaceCam и вызываем асинхронную функцию сравнения лиц
    await face_cam.compare_faces()

if __name__ == "__main__":
    # Создаем цикл событий и запускаем основную функцию
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
