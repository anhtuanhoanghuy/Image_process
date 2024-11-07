#!/usr/bin/env python

'''
Giải nén Wiener với kernel được ước lượng tự động từ ảnh ngắn và ảnh thường.

Sử dụng: python deconv_auto.py <hình ảnh ngắn> <hình ảnh thường>

Ví dụ:
  python deconv_auto.py new.png sample.png
'''

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.transform import resize

def load_images(short_path, regular_path):
    short_img = cv2.imread(short_path)
    regular_img = cv2.imread(regular_path)
    return short_img, regular_img

#Hàm Phát Hiện Điểm Đặc Trưng
def detect_feature_points(img, window_size=3, threshold=1000):
    # Tính toán cường độ cạnh ngang và dọc
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Tính toán cường độ cạnh
    edge_intensity_x = sobel_x ** 2
    edge_intensity_y = sobel_y ** 2

    # Lọc nhiễu bằng cách tích lũy các giá trị cường độ cạnh
    edge_intensity_x = cv2.GaussianBlur(edge_intensity_x, (window_size, window_size), 0)
    edge_intensity_y = cv2.GaussianBlur(edge_intensity_y, (window_size, window_size), 0)

    # Nhân các cường độ cạnh để tìm điểm đặc trưng
    feature_intensity = edge_intensity_x * edge_intensity_y

    # Chọn các điểm có giá trị cường độ cao là điểm đặc trưng
    feature_points = np.where(feature_intensity > threshold)
    return list(zip(feature_points[1], feature_points[0]))  # Trả về danh sách tọa độ (x, y)


def find_corresponding_points(short_img, regular_img, feature_points, window_size=8, max_points=10):
    corresponding_points = []
    h, w = regular_img.shape  # Kích thước ảnh dài

    # Giới hạn số lượng điểm đặc trưng để cải thiện hiệu suất
    feature_points = feature_points[:max_points]

    for (x, y) in feature_points:
        # Kiểm tra nếu vùng lân cận của điểm đặc trưng vượt ra ngoài biên của ảnh ngắn
        if y - window_size // 2 < 0 or y + window_size // 2 > short_img.shape[0] or \
                x - window_size // 2 < 0 or x + window_size // 2 > short_img.shape[1]:
            continue  # Bỏ qua điểm đặc trưng nếu nó vượt ra ngoài biên

        # Lấy vùng lân cận của điểm đặc trưng trong ảnh ngắn
        patch = short_img[y - window_size // 2: y + window_size // 2, x - window_size // 2: x + window_size // 2]

        # Kiểm tra nếu mẫu vượt quá biên của ảnh dài
        if window_size > h or window_size > w:
            continue  # Bỏ qua nếu mẫu lớn hơn ảnh dài

        # Sử dụng phương pháp ghép khối để tìm điểm tương ứng trong ảnh dài
        match = cv2.matchTemplate(regular_img, patch, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(match)

        corresponding_points.append((max_loc[0] + window_size // 2, max_loc[1] + window_size // 2))

    return corresponding_points

#Tính Toán Kernel Mờ Từ Các Cặp Điểm Tương Ứng
def estimate_blur_kernel(short_img, regular_img, feature_points, corresponding_points, kernel_size=64, gamma=0.01):
    m = len(feature_points)
    patches_s = []
    patches_r = []

    # Lấy các cặp điểm đặc trưng từ ảnh ngắn và ảnh dài
    for i in range(m):
        (x1, y1) = feature_points[i]
        (x2, y2) = corresponding_points[i]

        patch_s = short_img[y1 - kernel_size // 2: y1 + kernel_size // 2, x1 - kernel_size // 2: x1 + kernel_size // 2]
        patch_r = regular_img[y2 - kernel_size // 2: y2 + kernel_size // 2,
                  x2 - kernel_size // 2: x2 + kernel_size // 2]

        patches_s.append(fft2(patch_s))
        patches_r.append(fft2(patch_r))

    # Xây dựng hệ phương trình trong miền tần số và tính kernel mờ
    S = np.stack(patches_s)
    R = np.stack(patches_r)
    S_conj = np.conj(S)
    S_power = np.abs(S) ** 2

    # Tính kernel bằng bộ lọc Wiener
    K_est = (S_conj * R) / (S_power + gamma)
    kernel = np.real(ifft2(K_est.mean(axis=0)))
    kernel = fftshift(kernel)  # Dịch tâm kernel về giữa

    return kernel / kernel.sum()  # Chuẩn hóa kernel

def wiener_deconvolution(img, kernel, snr=0.01):
    # Chuyển ảnh và kernel sang miền Fourier
    img_fft = fft2(img)
    kernel_fft = fft2(kernel, s=img.shape)  # Kéo dài kernel cho kích thước của ảnh

    # Tính liên hợp phức của kernel
    kernel_conj = np.conj(kernel_fft)
    kernel_power = np.abs(kernel_fft) ** 2

    # Áp dụng bộ lọc Wiener
    result_fft = (img_fft * kernel_conj) / (kernel_power + snr)

    # Biến đổi Fourier ngược để lấy lại ảnh
    result = np.real(ifft2(result_fft))
    return result

def refine_blur_kernel(blur_kernel, iterations=1):
    """
    Tinh chỉnh kernel mờ bằng phương pháp đa phân giải và xử lý hình thái học.
    :param blur_kernel: Kernel mờ ban đầu đã được ước tính.
    :param iterations: Số lần thực hiện phép co và giãn.
    :return: Kernel mờ tinh chỉnh.
    """
    # Bước 1: Thu nhỏ kích thước kernel xuống còn một nửa
    half_size_kernel = resize(blur_kernel, (blur_kernel.shape[0] // 2, blur_kernel.shape[1] // 2), anti_aliasing=True)

    # Bước 2: Loại bỏ nhiễu bằng phép co và giãn
    for _ in range(iterations):
        half_size_kernel = binary_erosion(half_size_kernel).astype(float)
        half_size_kernel = binary_dilation(half_size_kernel).astype(float)

    # Bước 3: Phóng to kernel sau khi giảm nhiễu lên kích thước gốc
    refined_half_size_kernel = resize(half_size_kernel, blur_kernel.shape, anti_aliasing=True)

    # Bước 4: Tạo mặt nạ bằng phép AND logic với kernel gốc
    masked_kernel = np.logical_and(refined_half_size_kernel > 0.5, blur_kernel > 0.5).astype(float)

    # Bước 5: Áp dụng các phép co và giãn lần cuối trên kernel đã mặt nạ
    for _ in range(iterations):
        masked_kernel = binary_erosion(masked_kernel).astype(float)
        masked_kernel = binary_dilation(masked_kernel).astype(float)

    # Chuẩn hóa kernel tinh chỉnh để đảm bảo tổng của các phần tử bằng 1
    refined_kernel = masked_kernel / np.sum(masked_kernel)

    return refined_kernel
def main():
    # Đường dẫn tới ảnh short-exposure và regular-exposure
    short_path = "short_exposure2.png"
    regular_path = "long_exposure2.png"
    # Load images
    short_img, regular_img = load_images(short_path, regular_path)

    # Tách các kênh màu R, G, B
    short_b, short_g, short_r = cv2.split(short_img)
    regular_b, regular_g, regular_r = cv2.split(regular_img)
    # # Tạo ảnh rỗng cho kết quả sau khi giải nén từng kênh
    result_r = np.zeros_like(regular_r, dtype=np.float32)
    result_g = np.zeros_like(regular_g, dtype=np.float32)
    result_b = np.zeros_like(regular_b, dtype=np.float32)
    #
    # # Áp dụng quy trình cho từng kênh
    for (short_channel, regular_channel, result_channel) in zip(
            [short_r, short_g, short_b],
            [regular_r, regular_g, regular_b],
            [result_r, result_g, result_b]
    ):
    #     # Phát hiện điểm đặc trưng
        feature_points = detect_feature_points(short_channel)
        img_with_features = short_channel.copy()

        # Vẽ các điểm đặc trưng lên ảnh
        for (x, y) in feature_points:
            cv2.circle(img_with_features, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Vẽ một chấm đỏ

        # Hiển thị ảnh với các điểm đặc trưng
        cv2.imshow("Feature Points", img_with_features)
        # print(feature_points)
    #     # Tìm điểm tương ứng trong ảnh dài
        corresponding_points = find_corresponding_points(short_channel, regular_channel, feature_points)
        print(corresponding_points)
    #
    # if len(feature_points) != len(corresponding_points):
    #     print("Warning: Không đủ số lượng điểm tương ứng. Kiểm tra lại các tham số hoặc giảm số điểm đặc trưng.")
    #     return
    #     # Ước lượng và tinh chỉnh kernel mờ
    #     blur_kernel = estimate_blur_kernel(short_channel, regular_channel, feature_points, corresponding_points)
    #     refined_kernel = refine_blur_kernel(blur_kernel)
    #
    #     # Giải nén Wiener và lưu kết quả vào kênh tương ứng
    #     result_channel[:, :] = wiener_deconvolution(regular_channel, refined_kernel)
    #
    # # Kết hợp lại thành ảnh RGB
    # result_img = cv2.merge((result_b, result_g, result_r))

    # Hiển thị ảnh đã giải nén
    # cv2.imshow("Deconvolved Image", result_img / result_img.max())  # Chuẩn hóa để hiển thị
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()