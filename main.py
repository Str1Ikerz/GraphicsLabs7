import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import convolve2d

# Фильтр низких частот (размытие)
kernel_lowpass = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=float)
kernel_lowpass /= np.sum(kernel_lowpass)


# Лапласиан Гаусса (фильтр высоких частот)
def create_log_kernel(size=5, sigma=1.0):
    """Создание ядра лапласиана гауссиана"""
    kernel = np.fromfunction(
        lambda x, y: ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 - 2 * sigma ** 2) / sigma ** 4 *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel


kernel_log = create_log_kernel(5, 1.0)


def apply_filter(img_arr, kernel):
    """Применение свёрточного фильтра ко всем каналам изображения"""
    result = np.zeros_like(img_arr, dtype=float)
    for c in range(3):
        result[:, :, c] = convolve2d(img_arr[:, :, c].astype(float), kernel, mode='same', boundary='symm')
    return np.clip(result, 0, 255).astype(np.uint8)


def rgb_to_hsv(arr):
    """Конвертация RGB в HSV с использованием PIL (более надежный способ)"""
    img = Image.fromarray(arr.astype(np.uint8), 'RGB')
    hsv_img = img.convert('HSV')
    return np.array(hsv_img)


def hsv_to_rgb(arr):
    """Конвертация HSV в RGB с использованием PIL (более надежный способ)"""
    img = Image.fromarray(arr.astype(np.uint8), 'HSV')
    rgb_img = img.convert('RGB')
    return np.array(rgb_img)


def blur_bright_pixels(img_arr, threshold=200):
    """Размытие пикселей, превышающих порог яркости T"""
    hsv = rgb_to_hsv(img_arr)
    brightness = hsv[:, :, 2]  # V канал - яркость

    # Создаем маску пикселей с яркостью выше порога
    bright_mask = brightness > threshold

    # Применяем размытие только к ярким пикселям
    blurred = apply_filter(img_arr, kernel_lowpass)

    result = img_arr.copy()
    result[bright_mask] = blurred[bright_mask]

    return result


def apply_laplacian_of_gaussian_hsv(hsv_arr, channels):
    """Применение лапласиана гауссиана к указанным каналам HSV"""
    result = hsv_arr.copy().astype(float)

    for channel in channels:
        # Применяем лапласиан гауссиана к указанному каналу
        channel_data = hsv_arr[:, :, channel].astype(float)
        filtered_channel = convolve2d(channel_data, kernel_log, mode='same', boundary='symm')
        # Нормализуем и обрезаем значения
        filtered_channel = np.clip(filtered_channel, 0, 255)
        result[:, :, channel] = filtered_channel

    return result.astype(np.uint8)


def show_image_on(label, image_array, title):
    """Отобразить изображение в указанном Label"""
    if image_array is None:
        return
    img = Image.fromarray(image_array.astype(np.uint8))
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk
    label.master.children['title'].config(text=title)


def load_image():
    global img_array, img_original
    path = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg;*.png;*.bmp")])
    if not path:
        return
    img_original = Image.open(path).convert('RGB')
    img_array = np.array(img_original)
    show_image_on(label_original_img, img_array, "Исходное изображение")


def do_lowpass():
    global lowpass_img
    if img_array is None:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение!")
        return

    # Размытие пикселей, превышающих порог яркости T=200
    lowpass_img = blur_bright_pixels(img_array, threshold=200)
    show_image_on(label_low_img, lowpass_img, "Задание 1 (ФНЧ)")


def do_highpass():
    global highpass_img
    if img_array is None:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение!")
        return

    h, w, _ = img_array.shape
    hsv = rgb_to_hsv(img_array)

    # Разделяем изображение на левую и правую половины
    hsv_left = hsv[:, :w // 2].copy()
    hsv_right = hsv[:, w // 2:].copy()

    # Левая половина: лапласиан гауссиана для яркости (V) и насыщенности (S)
    # В HSV: H=0, S=1, V=2
    hsv_left_processed = apply_laplacian_of_gaussian_hsv(hsv_left, [1, 2])  # S и V каналы

    # Правая половина: лапласиан гауссиана для насыщенности (S) и цветового тона (H)
    hsv_right_processed = apply_laplacian_of_gaussian_hsv(hsv_right, [0, 1])  # H и S каналы

    # Объединяем обратно
    hsv_combined = np.hstack((hsv_left_processed, hsv_right_processed))
    highpass_img = hsv_to_rgb(hsv_combined)

    show_image_on(label_high_img, highpass_img, "Задание 2 (ФВЧ)")


def save_lowpass():
    if lowpass_img is None:
        messagebox.showwarning("Ошибка", "Сначала выполните задание 1!")
        return
    path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if path:
        Image.fromarray(lowpass_img).save(path)
        messagebox.showinfo("Сохранено", "Результат задания 1 сохранён!")


def save_highpass():
    if highpass_img is None:
        messagebox.showwarning("Ошибка", "Сначала выполните задание 2!")
        return
    path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if path:
        Image.fromarray(highpass_img).save(path)
        messagebox.showinfo("Сохранено", "Результат задания 2 сохранён!")


# Создание GUI
root = tk.Tk()
root.title("Фильтрация изображений — Вариант 13")
root.geometry("1000x700")
root.resizable(False, False)

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

btn_load = tk.Button(frame_buttons, text="Загрузить изображение", command=load_image, width=25)
btn_load.grid(row=0, column=0, padx=5)

btn_low = tk.Button(frame_buttons, text="Задание 1 (ФНЧ)", command=do_lowpass, width=18)
btn_low.grid(row=0, column=1, padx=5)

btn_high = tk.Button(frame_buttons, text="Задание 2 (ФВЧ)", command=do_highpass, width=18)
btn_high.grid(row=0, column=2, padx=5)

btn_save_low = tk.Button(frame_buttons, text="Сохранить результат 1", command=save_lowpass, width=20)
btn_save_low.grid(row=0, column=3, padx=5)

btn_save_high = tk.Button(frame_buttons, text="Сохранить результат 2", command=save_highpass, width=20)
btn_save_high.grid(row=0, column=4, padx=5)

frame_images = tk.Frame(root)
frame_images.pack(pady=10)


def make_image_frame(parent, title_text):
    frame = tk.Frame(parent, padx=10, pady=5)
    lbl_title = tk.Label(frame, text=title_text, name="title", font=("Arial", 12))
    lbl_title.pack()
    lbl_img = tk.Label(frame, bg="gray", width=300, height=300)
    lbl_img.pack()
    frame.pack(side=tk.LEFT, padx=10)
    return lbl_img


label_original_img = make_image_frame(frame_images, "Исходное изображение")
label_low_img = make_image_frame(frame_images, "Задание 1 (ФНЧ)")
label_high_img = make_image_frame(frame_images, "Задание 2 (ФВЧ)")

img_array = None
img_original = None
lowpass_img = None
highpass_img = None

root.mainloop()