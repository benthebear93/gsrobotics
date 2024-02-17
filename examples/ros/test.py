import numpy as np
import cv2
import math

# 이미지 생성 (높이, 너비, 채널)
img = np.zeros((240, 320, 3), dtype=np.uint8)

# 주어진 vectors 값
vectors = [180, 240, 135, 75, 86, 120]

# 주어진 초록색 점의 위치
green_point = [86, 120]

# 파란점과 빨간점 사이의 거리를 유지하면서, 초록색 점을 중심으로 양쪽에 배치
# 파란점과 빨간점의 y 좌표는 초록색 점과 같게 설정
# x 좌표의 거리 차이를 조정하여 파란점과 빨간점을 배치

# 거리 차이 설정 (x 좌표 기준)
distance_diff = 20

# 새로운 파란점과 빨간점의 위치 계산
# 파란점은 좌측에, 빨간점은 우측에 위치
blue_point_new = [green_point[0] - distance_diff, green_point[1]]
red_point_new = [green_point[0] + distance_diff, green_point[1]]

# 새로운 vectors 값
vectors = [red_point_new[0], red_point_new[1], blue_point_new[0], blue_point_new[1], green_point[0], green_point[1]]
vectors = [106, 120, 66, 120, 86, 120]
# 변환 상수
RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180
mm2m = 1/1000

# 계산된 목표 위치
arrow_length = 50
goal_theta = 135 * DEG2RAD  # 라디안으로 변환
goal_y = arrow_length * math.cos(math.pi - goal_theta)
goal_x = arrow_length * math.sin(math.pi - goal_theta)

# 원 그리기
cv2.circle(img, (vectors[0], vectors[1]), 5, (0, 0, 255), -1)  # 빨간색 큰 X
cv2.circle(img, (vectors[2], vectors[3]), 5, (255, 0, 0), -1)  # 파란색 작은 X
cv2.circle(img, (vectors[4], vectors[5]), 5, (0, 255, 0), -1)  # 초록색 중심점

# 목표 위치에 원 그리기 및 선 그리기
cv2.circle(img, (int(vectors[4] + 2*goal_x), int(vectors[5] + 2*goal_y)), 5, (255, 255, 255), -1)  # 흰색 목표점
cv2.line(img, (int(vectors[4] + 2*goal_x), int(vectors[5] + 2*goal_y)), (int(vectors[4]), int(vectors[5])), (255, 255, 255), thickness=2)

# 각도 계산
smallx = [vectors[2], vectors[3]]
bigx = [vectors[4], vectors[5]]
if smallx[1] > bigx[1]:  # theta2
    theta = math.pi - math.atan2(smallx[1] - bigx[1], bigx[0] - smallx[0])
else:  # theta1
    theta = math.atan2(bigx[1] - smallx[1], bigx[0] - smallx[0])

# 센서 데이터 출력 및 회전각도 출력
print("Sensor data:", theta * RAD2DEG)

# 이미지 표시
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
