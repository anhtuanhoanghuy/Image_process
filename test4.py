import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def main():
    short_path = "short3.png"
    regular_path = "long3.png"

    # Đọc ảnh
    short_img = cv2.imread(short_path, cv2.IMREAD_GRAYSCALE)
    regular_img = cv2.imread(regular_path, cv2.IMREAD_GRAYSCALE)

    # Phát hiện các điểm đặc trưng và tính toán Homography
    homography, img_matches = detect_keypoints_and_compute_kernel(short_img, regular_img)

    # Hiển thị kết quả khớp các điểm đặc trưng
    plt.imshow(img_matches)
    plt.title('Feature Matches between Blurred and Original Images')
    plt.show()

    # In ra ma trận đồng nhất
    print("Homography Matrix (Homography to estimate the blur kernel):")
    print(homography)


if __name__ == "__main__":
    main()
