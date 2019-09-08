import numpy as np
import scipy as scipy
from numpy.random import uniform    # 均匀分布
import scipy.stats

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2


x_range = np.array([0, 800])
y_range = np.array([0, 600])

# Number of partciles
N = 400

landmarks = np.array([[144, 73], [410, 13], [336, 175], [718, 159], [178, 484], [665, 464]])
NL = len(landmarks)


weights = np.array([1.0] * N)

center = np.array([[-10, -10]])

trajectory = np.zeros(shape=(0, 2))
robot_pos = np.zeros(shape=(0, 2))
previous_x = -1
previous_y = -1
DELAY_MSEC = 50

WIDTH = 800
HEIGHT = 600
WINDOW_NAME = "Particle Filter"

# sensorMu=0
# sensorSigma=3

sensor_std_err = 5



def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    # cv2版本3以上使用LINE_AA属性
    LINE_AA = cv2.LINE_AA if cv2.__version__[0] >= '3' else cv2.CV_AA
    color = (b, g, r)  # cv2的颜色参数是BGR模式？？？
    ctrx = center[0, 0]
    ctry = center[0, 1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)


def mouseCallback(event, x, y, flags, null):
    global center
    global trajectory
    global previous_x
    global previous_y
    global zs

    center = np.array([[x, y]])
    trajectory = np.vstack((trajectory, np.array([x, y])))
    # noise=sensorSigma * np.random.randn(1,2) + sensorMu

    if previous_x > 0:
        heading = np.arctan2(np.array([y - previous_y]), np.array([previous_x - x]))

        if heading > 0:
            heading = -(heading - np.pi)
        else:
            heading = -(np.pi + heading)

        distance = np.linalg.norm(np.array([[previous_x, previous_y]]) - np.array([[x, y]]), axis=1)

        std = np.array([2, 4])
        u = np.array([heading, distance])
        predict(particles, u, std, dt=1.)

        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
        update_by_poisson(particles, weights, z=zs, landmarks=landmarks)

        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)

    previous_x = x
    previous_y = y


# 创建初始化粒子集合，使用均匀分布
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles


# 预测函数，根据当前粒子集合particles和运动指令u，加上噪声预测下一时刻的粒子集合
def predict(particles, u, std, dt=1.):
    N = len(particles)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist


# 使用泊松分布计算更新权重
def update_by_poisson(particles, weights, z, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.power((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2, 0.5)
        weights *= scipy.stats.poisson.pmf(z[i], 50, distance)

    weights += 1.e-300
    weights /= sum(weights)


# 为避免粒子数越来越少要对粒子进行重采样，这里采用的是低方差非独立随机重采样
def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N  # 划分区间

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)  # 对权值进行累加
    i, j = 0, 0  # i代表重采样个数，j代表粒子下标
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


# 根据重采样得到的下标重构粒子集合
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)




if __name__ == '__main__':

    particles = create_uniform_particles(x_range, y_range, N)
    # Create a black image, a window and bind the function to window
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

    while (1):

        cv2.imshow(WINDOW_NAME, img)
        img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        drawLines(img, trajectory, 0, 255, 0)
        drawCross(img, center, r=255, g=0, b=0)

        # landmarks
        for landmark in landmarks:
            cv2.circle(img, tuple(landmark), 10, (255, 0, 0), -1)

        # draw_particles:
        for particle in particles:
            cv2.circle(img, tuple((int(particle[0]), int(particle[1]))), 1, (255, 255, 255), -1)

        if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
            break

        cv2.circle(img, (10, 10), 10, (255, 0, 0), -1)
        cv2.circle(img, (10, 30), 3, (255, 255, 255), -1)
        cv2.putText(img, "Landmarks", (30, 20), 1, 1.0, (255, 0, 0))
        cv2.putText(img, "Particles", (30, 40), 1, 1.0, (255, 255, 255))
        cv2.putText(img, "Robot Trajectory(Ground truth)", (30, 60), 1, 1.0, (0, 255, 0))

        drawLines(img, np.array([[10, 55], [25, 55]]), 0, 255, 0)

    cv2.destroyAllWindows()