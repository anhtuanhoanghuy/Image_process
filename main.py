import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from skimage.transform import resize
import matplotlib.pyplot as plt
def load_images(short_path, regular_path):
    short_img = cv2.imread(short_path, cv2.IMREAD_GRAYSCALE)
    regular_img = cv2.imread(regular_path, cv2.IMREAD_GRAYSCALE)
    regular_rgb = cv2.imread(regular_path)
    return short_img, regular_img, regular_rgb
def detect_keypoints_and_compute_kernel(short_img, regular_img):
    # Phát hiện điểm đặc trưng sử dụng ORB
    orb = cv2.ORB_create()

    # Tìm các điểm đặc trưng và mô tả
    kp1, des1 = orb.detectAndCompute(short_img, None)  # Ảnh bị mờ
    kp2, des2 = orb.detectAndCompute(regular_img, None)  # Ảnh gốc

    # Khớp các điểm đặc trưng giữa ảnh bị mờ và ảnh gốc
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sắp xếp các cặp điểm đặc trưng theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)

    # Vẽ các điểm đặc trưng và các đường nối giữa chúng
    img_matches = cv2.drawMatches(short_img, kp1, regular_img, kp2, matches[:10], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Tính toán Homography từ các điểm đặc trưng đã khớp
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Tính toán ma trận Homography (tương quan hình học)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M, img_matches

def estimate_blur_kernel(short_img, regular_img, gamma=0.01):
    S = fft2(short_img)
    R = fft2(regular_img)
    S_conj = np.conj(S)
    S_power = np.abs(S) ** 2
    K_est = (S_conj * R) / (S_power + gamma)
    k = np.real(ifft2(K_est))
    k = fftshift(k)
    return k / k.sum()

def resize_blur_kernel(origin_img, scale_factor):
    new_shape = (int(origin_img.shape[0] * scale_factor), int(origin_img.shape[1] * scale_factor))
    resize_kernel = resize(origin_img, new_shape, anti_aliasing=True)
    return resize_kernel

def erosion(img, iterations=1):
    k = np.ones((2, 2), np.uint8)
    erosion_img = cv2.erode(img, k, iterations=iterations)
    return erosion_img

def dilation(img, iterations=1):
    k = np.ones((2, 2), np.uint8)
    dilation_img = cv2.dilate(img, k, iterations=iterations)
    return dilation_img

def AND_image(blur_kernel, unblur_kernel):
    # Điều chỉnh kích thước của mảng nhỏ hơn để khớp với mảng lớn hơn
    min_shape = np.minimum(blur_kernel.shape, unblur_kernel.shape)
    # Cắt kích thước của cả hai mảng về kích thước nhỏ hơn
    blur_kernel = blur_kernel[:min_shape[0], :min_shape[1]]
    unblur_kernel = unblur_kernel[:min_shape[0], :min_shape[1]]
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
    result_fft = (img_fft * kernel_conj) / (kernel_power + snr)
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
def main():
    short_path = "short3.png"
    regular_path = "long3.png"
    short_img, regular_img, regular_rgb = load_images(short_path, regular_path)
    kernel = estimate_blur_kernel(short_img, regular_img)
    half_size_kernel = resize_blur_kernel(kernel, 0.5)
    half_size_kernel = erosion(half_size_kernel)
    half_size_kernel = dilation(half_size_kernel)
    resize_kernel = resize_blur_kernel(half_size_kernel, 2)
    and_kernel = AND_image(kernel, resize_kernel)
    result_kernel = erosion(and_kernel)
    result_kernel = dilation(result_kernel)
    result_img = wiener_deconvolution2(regular_img, kernel)
    result_rgb = process_rgb_channels(regular_rgb, kernel)
    # Normalize images for display with OpenCV
    result_img_norm = normalize_image(result_img).astype(np.uint8)
    result_rgb_norm = np.dstack([normalize_image(result_rgb[:, :, i]) for i in range(3)]).astype(np.uint8)
    # Phát hiện các điểm đặc trưng và tính toán Homography
    homography, img_matches = detect_keypoints_and_compute_kernel(short_img, regular_img)
    cv2.imshow("Origin Long_exposure Image", regular_rgb)
    cv2.imshow("Origin Blur Kernel", kernel / kernel.max())
    cv2.imshow("Result Kernel", result_kernel / result_kernel.max())
    cv2.imshow("Result Image", result_rgb_norm)
    # Hiển thị kết quả khớp các điểm đặc trưng
    plt.imshow(img_matches)
    plt.title('Feature Matches between Blurred and Original Images')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
