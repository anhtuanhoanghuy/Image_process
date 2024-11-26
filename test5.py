import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
from skimage.transform import resize
import matplotlib.pyplot as plt

def load_images(short_path, regular_path):
    # Đọc ảnh short-exposure và regular-exposure
    short_img = cv2.imread(short_path, cv2.IMREAD_GRAYSCALE)
    regular_img = cv2.imread(regular_path, cv2.IMREAD_GRAYSCALE)
    regular_rgb = cv2.imread(regular_path)
    return short_img, regular_img, regular_rgb

def detect_and_match_keypoints_with_kernels(short_img, regular_img, patch_size=64, gamma=0.01):
    # Phát hiện điểm đặc trưng và khớp chúng
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(short_img, None)
    kp2, des2 = orb.detectAndCompute(regular_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Lấy tọa độ các điểm đặc trưng khớp
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Cắt các vùng đặc trưng
    src_patches = []
    dst_patches = []
    for pt_src, pt_dst in zip(src_pts, dst_pts):
        x_src, y_src = int(pt_src[0]), int(pt_src[1])
        x_dst, y_dst = int(pt_dst[0]), int(pt_dst[1])
        if (y_src - patch_size // 2 >= 0 and y_src + patch_size // 2 < short_img.shape[0] and
            x_src - patch_size // 2 >= 0 and x_src + patch_size // 2 < short_img.shape[1] and
            y_dst - patch_size // 2 >= 0 and y_dst + patch_size // 2 < regular_img.shape[0] and
            x_dst - patch_size // 2 >= 0 and x_dst + patch_size // 2 < regular_img.shape[1]):
            src_patches.append(short_img[y_src - patch_size // 2:y_src + patch_size // 2,
                                         x_src - patch_size // 2:x_src + patch_size // 2])
            dst_patches.append(regular_img[y_dst - patch_size // 2:y_dst + patch_size // 2,
                                           x_dst - patch_size // 2:x_dst + patch_size // 2])

    # Tính các kernel từ các patch
    kernels = compute_kernels_from_patches(src_patches, dst_patches, gamma)

    # Tổng hợp kernel cuối cùng
    final_kernel = combine_kernels(kernels)

    # Tính toán Homography từ các điểm đặc trưng
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M, final_kernel, kernels, src_patches, dst_patches, matches

def estimate_blur_kernel(short_img, regular_img, gamma=0.01):
    # Biến đổi Fourier và tính kernel
    S = np.fft.fft2(short_img)
    R = np.fft.fft2(regular_img)
    S_conj = np.conj(S)
    S_power = np.abs(S) ** 2
    K_est = (S_conj * R) / (S_power + gamma)
    k = np.real(ifft2(K_est))
    k = fftshift(k)
    return k / k.sum()
def compute_kernels_from_patches(src_patches, dst_patches, gamma=0.01):
    kernels = []  # Danh sách lưu các kernel từ từng cặp patch
    for src_patch, dst_patch in zip(src_patches, dst_patches):
        # Kiểm tra kích thước vùng có hợp lệ không
        if src_patch.shape == dst_patch.shape and src_patch.size > 0:
            # Tính kernel cho từng cặp patch
            kernel = estimate_blur_kernel(src_patch, dst_patch, gamma)
            kernels.append(kernel)
    return kernels

def combine_kernels(kernels):
    # Tổng hợp kernel bằng cách lấy trung bình
    if len(kernels) == 0:
        raise ValueError("Không có kernel nào được tính toán!")
    final_kernel = np.mean(kernels, axis=0)
    return final_kernel / np.sum(final_kernel)  # Chuẩn hóa kernel

def refine_blur_kernel(kernel, iterations=1):
    # Refinement bằng erosion, dilation và AND logic
    kernel_resized = resize_blur_kernel(kernel, 0.5)
    kernel_refined = erosion(kernel_resized, iterations=iterations)
    kernel_refined = dilation(kernel_refined, iterations=iterations)
    kernel_resized_back = resize_blur_kernel(kernel_refined, 2)
    refined_kernel = AND_image(kernel, kernel_resized_back)
    return refined_kernel

def resize_blur_kernel(origin_img, scale_factor):
    new_shape = (int(origin_img.shape[0] * scale_factor), int(origin_img.shape[1] * scale_factor))
    return resize(origin_img, new_shape, anti_aliasing=True)

def erosion(img, iterations=1):
    k = np.ones((2, 2), np.uint8)
    return cv2.erode(img, k, iterations=iterations)

def dilation(img, iterations=1):
    k = np.ones((2, 2), np.uint8)
    return cv2.dilate(img, k, iterations=iterations)

def AND_image(blur_kernel, unblur_kernel):
    mask1 = blur_kernel > 0
    mask2 = unblur_kernel > 0
    and_mask = np.logical_and(mask1, mask2)
    and_result = np.where(and_mask, blur_kernel, 0)
    return and_result / np.sum(and_result)

def wiener_deconvolution2(img, kernel, snr=0.01):
    img_fft = fft2(img)
    kernel_fft = fft2(kernel, s=img.shape)
    kernel_conj = np.conj(kernel_fft)
    kernel_power = np.abs(kernel_fft) ** 2
    result_fft = (img_fft * kernel_conj) / (kernel_power + snr)
    result = np.real(ifft2(result_fft))
    return fftshift(result)

def process_rgb_channels(regular_rgb, kernel):
    result_rgb = np.zeros_like(regular_rgb, dtype=np.float32)
    for i in range(3):
        result_rgb[:, :, i] = wiener_deconvolution2(regular_rgb[:, :, i].astype(np.float32), kernel)
    return result_rgb

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min()) * 255

def main():
    short_path = "short3.png"
    regular_path = "long3.png"

    # Load images
    short_img, regular_img, regular_rgb = load_images(short_path, regular_path)

    # Detect and match feature points and Estimate the blur kernel
    M, final_kernel, kernels, src_patches, dst_patches, matches = detect_and_match_keypoints_with_kernels(
        short_img, regular_img
    )
    # # Estimate the blur kernel

    half_size_kernel = resize_blur_kernel(final_kernel, 0.5)
    half_size_kernel = erosion(half_size_kernel)
    half_size_kernel = dilation(half_size_kernel)
    resize_kernel = resize_blur_kernel(half_size_kernel, 2)
    and_kernel = AND_image(final_kernel, resize_kernel)
    result_kernel = erosion(and_kernel)
    result_kernel = dilation(result_kernel)
    result_img = wiener_deconvolution2(regular_img, final_kernel)
    result_rgb = process_rgb_channels(regular_rgb, final_kernel)

    # Normalize images for display
    result_img_norm = normalize_image(result_img).astype(np.uint8)
    result_rgb_norm = np.dstack([normalize_image(result_rgb[:, :, i]) for i in range(3)]).astype(np.uint8)

    # Display results
    cv2.imshow("Origin image", regular_img / result_img.max())
    cv2.imshow("RGB Image", regular_rgb)
    cv2.imshow("Origin Blur Kernel", final_kernel / final_kernel.max())
    # cv2.imshow("Estimated Blur Kernel", half_size_kernel / half_size_kernel.max())
    # cv2.imshow("Resize", resize_kernel / resize_kernel.max())
    # cv2.imshow("Result Kernel", result_kernel / result_kernel.max())
    cv2.imshow("Deconvolved Image", result_img_norm)
    cv2.imshow("Result RGB", result_rgb_norm)
    # Hiển thị kết quả khớp các điểm đặc trưng
    # plt.imshow(matches)
    # plt.title('Feature Matches between Blurred and Original Images')
    # plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()