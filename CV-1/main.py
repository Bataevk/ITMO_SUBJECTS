# sobel_filter.py
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

def sobel_native(img_gray):
    """Нативная реализация с NumPy sliding_window_view"""
    height, width = img_gray.shape
    img = img_gray.astype(np.float32)
    
    # Создаем отступы 1 по краям
    padded_image = np.pad(img, ((1, 1), (1, 1)), mode='constant')

    # Создаём sliding window view: shape (height, width, 3, 3)
    window = sliding_window_view(padded_image, (3, 3))
    
    # Ядра Собеля
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Векторизованная свёртка: умножаем и суммируем по осям 2 и 3
    gx = np.sum(window * kernel_x[None, None, :, :], axis=(2, 3))
    gy = np.sum(window * kernel_y[None, None, :, :], axis=(2, 3))
    
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Создаём полный результат
    result = magnitude
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def sobel_opencv(img_gray):
    """Реализация через встроенные функции OpenCV"""
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1,0,ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F,0,1,ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


def main():
    # Загружаем изображение
    img = cv2.imread('image.jpg')
    if img is None:
        print("Ошибка: изображение image.jpg не найдено!")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Тестируем быстродействие
    iterations = 50
    
    start = time.time()
    for _ in range(iterations):
        _ = sobel_opencv(gray)
    time_opencv = (time.time() - start) / iterations
    
    start = time.time()
    for _ in range(iterations):
        _ = sobel_native(gray)
    time_native = (time.time() - start) / iterations
    
    print(f"OpenCV:     {time_opencv*1000:.3f} мс")
    print(f"Нативный:   {time_native*1000:.3f} мс")
    print(f"Ускорение:  {time_native/time_opencv:.1f}x")
    
    # Получаем результаты один раз для сохранения
    result_cv = sobel_opencv(gray)
    result_nat = sobel_native(gray)
    
    # Сохраняем изображения
    cv2.imwrite('result_opencv.jpg', result_cv)
    cv2.imwrite('result_native.jpg', result_nat)
    
    # Показываем сравнение
    plt.figure(figsize=(15,5))
    plt.subplot(131); plt.imshow(gray, cmap='gray'); plt.title('Оригинал')
    plt.subplot(132); plt.imshow(result_cv, cmap='gray'); plt.title('OpenCV')
    plt.subplot(133); plt.imshow(result_nat, cmap='gray'); plt.title('Нативный Python')
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()