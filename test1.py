import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from skimage.transform import resize
import matplotlib.pyplot as plt

def load_images(short_path, regular_path):
    short_img = cv2.imread(short_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    regular_img = cv2.imread(regular_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    regular_rgb = cv2.imread(regular_path)
    return short_img, regular_img, regular_rgb

def estimate_blur_kernel(short_img, regular_img, gamma=0.01):
    S = fft2(short_img)
    R = fft2(regular_img)
    S_conj = np.conj(S)
    S_power = np.abs(S) ** 2
    # K_est = (S_conj * R) / (S_power + gamma)
    K_est = (S_conj * R) / (S * S_conj + gamma)
    k = np.real(ifft2(K_est))
    k = fftshift(k)
    return k / k.sum()

def resize_blur_kernel(origin_img, scale_factor):
    new_shape = (int(origin_img.shape[0] * scale_factor), int(origin_img.shape[1] * scale_factor))
    resize_kernel = resize(origin_img, new_shape, anti_aliasing=True)
    return resize_kernel

def erosion(img, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    kernel = np.array([[0, 1, 0],
                   [1, 1, 0],
                   [0, 1, 0]], np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))
    erosion_img = cv2.erode(img, kernel, iterations=iterations)
    return erosion_img

def dilation(img, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))
    dilation_img = cv2.dilate(img, kernel, iterations=iterations)
    # performing opening using its function
    # opening = cv2.morphologyEx(dilation_img, cv2.MORPH_OPEN, kernel)
    # performing closing using its function
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return dilation_img
    # return opening
    # return closing

def AND_image(blur_kernel, unblur_kernel):
    mask1 = blur_kernel > 0
    mask2 = unblur_kernel > 0
    and_mask = np.logical_and(mask1, mask2)
    and_result = np.where(and_mask, blur_kernel, 0)
    kernel_sum = np.sum(and_result)
    result_kernel = and_result / kernel_sum
    return result_kernel

def wiener_deconvolution2(img, kernel, snr=0.01):
    # Chuyển ảnh và kernel sang miền Fourier
    img_fft = fft2(img)
    kernel_fft = fft2(kernel, s=img.shape) # Kéo dài kernel cho kích thước của ảnh
    # Tính liên hợp phức của kernel
    kernel_conj = np.conj(kernel_fft)
    kernel_power = np.abs(kernel_fft) ** 2
    # Áp dụng bộ lọc Wiener
    result_fft = (img_fft * kernel_conj) / (kernel_fft * kernel_conj + snr)
    # Biến đổi Fourier ngược để lấy lại ảnh
    result = np.real(ifft2(result_fft))
    np.real(ifft2(result_fft))
    # Dùng fftshift để dịch tâm về giữa
    result = fftshift(result)
    return result

def process_rgb_channels(regular_rgb, kernel):
    result_rgb = np.zeros_like(regular_rgb, dtype=np.float32)
    for i in range(3):  # Process each color channel separately
        result_rgb[:, :, i] = wiener_deconvolution2(regular_rgb[:, :, i].astype(np.float32), kernel)
    return result_rgb

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min()) * 255


def display_image():
    # Display results

    display = [
        regular_img / result_img.max(),
        regular_rgb,
        kernel / kernel.max(),
        wiener_img,
        resize_blur_kernel(kernel, 0.5),
        resize_kernel,
        result_kernel / result_kernel.max(),
        result_img_norm,
    ]

    label = [
        'Original Image',
        'RGB Image',
        'Origin Blur Kernel',
        'Wiener Image',
        'Half size',
        'Estimated Blur Kernel',
        'Result Kernel',
        'Deconvolved Image',
    ]

    # Reduced figure size
    fig = plt.figure(figsize=(8, 6))  # Adjusted size

    for i in range(len(display)):
        fig.add_subplot(2, 4, i + 1)
        
        # Apply 'gray' colormap only if image is not RGB
        if display[i].ndim == 2:  # Assuming RGB images have 3 dimensions
            plt.imshow(display[i], cmap='gray')
        else:
            plt.imshow(display[i])

        plt.title(label[i], fontsize=8)  # Smaller font size for titles
        plt.axis('off')  # Optional: remove axes for a cleaner look

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    short_path = "short.png"
    regular_path = "long.png"
    short_img, regular_img, regular_rgb = load_images(short_path, regular_path)
    kernel = estimate_blur_kernel(short_img, regular_img)
    wiener_img = wiener_deconvolution2(regular_img, kernel)
    half_size_kernel = resize_blur_kernel(kernel, 0.5)
    half_size_kernel = erosion(half_size_kernel)
    half_size_kernel = dilation(half_size_kernel)
    resize_kernel = resize_blur_kernel(half_size_kernel, 2)
    and_kernel = AND_image(kernel, resize_kernel)
    result_kernel = erosion(and_kernel)
    result_kernel = dilation(result_kernel)
    result_img = wiener_deconvolution2(regular_img, result_kernel)
    result_rgb = process_rgb_channels(regular_rgb, result_kernel)

    # Normalize images for display with OpenCV
    result_img_norm = normalize_image(result_img).astype(np.uint8)
    result_rgb_norm = np.dstack([normalize_image(result_rgb[:, :, i]) for i in range(3)]).astype(np.uint8)
    # print(result_img)
    display_image()
