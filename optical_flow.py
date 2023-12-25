import cv2
import numpy as np


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_brg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(img_brg, lines, 0, (0, 255, 0))

    for(x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_brg, (x1, y1), 1, (0, 255, 0), -1)

    return img_brg


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi

    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v*4, 255)

    brg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return brg
