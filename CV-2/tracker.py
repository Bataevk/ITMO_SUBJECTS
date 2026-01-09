import cv2
import numpy as np
import sys
import os

def track_object(video_path, output_path=None):
    if not os.path.exists(video_path):
        print(f"Ошибка: Видео не найдено по пути {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    # Считываем первый кадр
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось считать первый кадр.")
        return

    # Инициализация SIFT
    sift = cv2.SIFT_create()

    # Находим ключевые точки и дескрипторы на первом кадре
    kp1, des1 = sift.detectAndCompute(first_frame, None)
    
    h_obj, w_obj = first_frame.shape[:2]
    # Углы объекта (весь первый кадр)
    obj_corners = np.float32([[0, 0], [0, h_obj-1], [w_obj-1, h_obj-1], [w_obj-1, 0]]).reshape(-1, 1, 2)

    # Параметры FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Настройка записи видео, если указан путь для сохранения
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Записываем первый кадр с рамкой
    if out:
        first_frame_disp = first_frame.copy()
        cv2.polylines(first_frame_disp, [np.int32(obj_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(first_frame_disp, "Объект", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        out.write(first_frame_disp)

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Находим ключевые точки и дескрипторы на текущем кадре
        kp2, des2 = sift.detectAndCompute(frame, None)

        if des2 is not None and len(des2) > 0:
            # Сопоставляем дескрипторы
            matches = flann.knnMatch(des1, des2, k=2)

            # Отбираем хорошие совпадения
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # Если достаточно совпадений, ищем объект
            MIN_MATCH_COUNT = 10
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Находим гомографию
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Проецируем углы объекта на текущий кадр
                    dst_corners = cv2.perspectiveTransform(obj_corners, M)
                    
                    # Рисуем рамку
                    frame = cv2.polylines(frame, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # Добавляем подпись
                    x, y = np.int32(dst_corners[0][0])
                    cv2.putText(frame, "Tracked Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if out:
            out.write(frame)
        
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Обработано {frame_count} кадров. Результат сохранен в {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python tracker.py <путь_к_видео> [путь_к_выходному_видео]")
    else:
        video = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "output.avi"
        
        # Проверяем, существует ли директория
        out_dir = os.path.dirname(output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        track_object(video, output)

