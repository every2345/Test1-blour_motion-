import cv2
import cvzone
import numpy as np
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)  # Mở camera

if not cap.isOpened():  # Kiểm tra camera có được mở hay không
    print("Không thể mở camera")
    exit()

# Khởi tạo PoseDetector
detector = PoseDetector()
fpsReader = cvzone.FPS()

while True:
    ret, frame1 = cap.read()  # Đọc khung hình hiện tại
    ret, frame2 = cap.read()  # Đọc khung hình tiếp theo

    if not ret:  # Kiểm tra khung hình rỗng
        print("Khung hình rỗng")
        break

    frame1 = detector.findPose(frame1)  # Dò tìm vị trí các điểm keypoint trong khung hình
    lmlist, bboxInfo = detector.findPosition(frame1, bboxWithHands=True)  # Lấy thông tin về vị trí các điểm keypoint và bounding box

    fps, frame1 = fpsReader.update(frame1, pos=(10, 30), color=(0, 255, 0), scale=2, thickness=3)  # Hiển thị FPS

    frameDiff = cv2.absdiff(frame1, frame2)  # Tính toán khung hình khác biệt giữa hai khung hình liên tiếp
    grayDiff = cv2.cvtColor(frameDiff, cv2.COLOR_BGR2GRAY)  # Chuyển khung hình khác biệt sang ảnh xám
    _, colorDiff = cv2.threshold(grayDiff, 50, 255, cv2.THRESH_BINARY)  # Chuyển ảnh xám thành ảnh nhị phân
    motion_pixels = cv2.countNonZero(colorDiff)  # Đếm số pixel có chuyển động

    if motion_pixels > 23000:
        print("Hành động té ngã được phát hiện với", motion_pixels, "pixel.")

    # Tính toán tỷ lệ giữa chiều rộng và chiều cao của khung hình
    frame_ratio = frame1.shape[1] / frame1.shape[0]
    new_width = 800  # Định nghĩa chiều rộng mới của khung hình

    # Tính toán chiều cao mới dựa trên tỷ lệ và chiều rộng mới
    new_height = int(new_width / frame_ratio)

    # Thay đổi kích thước khung hình
    frame1 = cv2.resize(frame1, (new_width, new_height))

    cv2.imshow("Original - camera", frame1)  # Hiển thị khung hình gốc
    cv2.imshow("Pixel", colorDiff)  # Hiển thị khung hình chuyển động

    if cv2.waitKey(1) == 27:  # Thoát nếu nhấn phím ESC
        break

cap.release()
cv2.destroyAllWindows()
