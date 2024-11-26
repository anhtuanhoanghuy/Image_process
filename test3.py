import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from skimage.transform import resize
def detect_feature_points(image):
    # Phát hiện các điểm đặc trưng sử dụng ORB
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
def match_feature_points(descriptors_short, descriptors_long):
    # Sử dụng bộ đối sánh BFMatcher để tìm các điểm tương đồng
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_short, descriptors_long)
    # Sắp xếp các điểm tương đồng theo độ chính xác
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def estimate_blur_kernel(matches, keypoints_short, keypoints_long, kernel_size=(15, 15)):
    # Tạo ma trận để lưu trữ chuyển động mờ
    motion_vectors = []

    for match in matches:
        # Lấy tọa độ các điểm tương đồng từ cặp ảnh
        pt_short = np.array(keypoints_short[match.queryIdx].pt)
        pt_long = np.array(keypoints_long[match.trainIdx].pt)

        # Tính toán vector chuyển động giữa điểm tương đồng trong ảnh ngắn và dài
        motion_vector = pt_long - pt_short
        motion_vectors.append(motion_vector)

    # Tính trung bình của các vector chuyển động để xác định độ dài và hướng hạt nhân mờ
    avg_motion = np.mean(motion_vectors, axis=0)

    # Khởi tạo hạt nhân mờ (blur kernel) dựa trên trung bình vector chuyển động
    kernel = np.zeros(kernel_size, dtype=np.float32)
    # Đặt giá trị cho hạt nhân mờ theo hướng và độ dài của chuyển động
    center = (kernel_size[0] // 2, kernel_size[1] // 2)
    end_point = (int(center[0] + avg_motion[0]), int(center[1] + avg_motion[1]))
    cv2.line(kernel, center, end_point, 1, thickness=1)

    # Chuẩn hóa kernel để tổng của các phần tử bằng 1
    kernel /= kernel.sum()
    return kernel


def main(short_exposure_image, long_exposure_image):
    # Bước 1: Phát hiện điểm đặc trưng trong cả hai ảnh
    keypoints_short, descriptors_short = detect_feature_points(short_exposure_image)
    keypoints_long, descriptors_long = detect_feature_points(long_exposure_image)

    # Bước 2: Đối sánh các điểm đặc trưng
    matches = match_feature_points(descriptors_short, descriptors_long)

    # Bước 3: Ước lượng hạt nhân mờ
    blur_kernel = estimate_blur_kernel(matches, keypoints_short, keypoints_long)

    return blur_kernel

# Hàm hiển thị blur kernel
def display_blur_kernel(blur_kernel):
    plt.imshow(blur_kernel, cmap='gray')
    plt.colorbar()
    plt.title("Estimated Blur Kernel")
    plt.show()
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
# Đọc ảnh phơi sáng ngắn và dài

def process_rgb_channels(regular_rgb, kernel):
    result_rgb = np.zeros_like(regular_rgb, dtype=np.float32)
    for i in range(3):  # Process each color channel separately
        result_rgb[:, :, i] = wiener_deconvolution2(regular_rgb[:, :, i].astype(np.float32), kernel)
    return result_rgb

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min()) * 255
# Tính toán hạt nhân mờ
short_exposure_image = cv2.imread('new.png', cv2.IMREAD_GRAYSCALE)
long_exposure_image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
long_rgb = cv2.imread('sample.png')
blur_kernel = main(short_exposure_image, long_exposure_image)
result_img = wiener_deconvolution2(long_exposure_image, blur_kernel)
result_rgb = process_rgb_channels(long_rgb, blur_kernel)

# Normalize images for display with OpenCV
result_img_norm = normalize_image(result_img).astype(np.uint8)
result_rgb_norm = np.dstack([normalize_image(result_rgb[:, :, i]) for i in range(3)]).astype(np.uint8)
cv2.imshow("Result ", result_img)
display_blur_kernel(blur_kernel)
# cv2.imshow("Deconvolved Image", result_img_norm)
cv2.imshow("Result RGB", result_rgb_norm)
# print("Hạt nhân mờ đã ước lượng:")
# print(blur_kernel)