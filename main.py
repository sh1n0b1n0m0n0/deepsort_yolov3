import numpy as np
import pandas as pd
from pathlib import Path
import tracker
from detector import Detector
from inexlib.object_detection.yolo_detector import YoloDarknet
from inexlib.utils.image_sequence import create_from_file_system
import cv2
import os
from optical_flow import draw_flow, draw_hsv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d' % (x, y))


def count_elems(lst):
    lst.sort()
    return {f'id_{elem}': lst.count(elem) for elem in lst}


def dump_trackers(trackers_info, video_path):
    df = pd.DataFrame.from_dict(trackers_info)
    df.to_csv(Path(f'trackers_{Path(video_path).name}.csv'), index=False)


def main():
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                     [299, 375], [267, 289]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                       [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))
    blue_color_plate = [255, 0, 0]
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
    yellow_color_plate = [0, 255, 255]
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)
    color_polygons_image = blue_image + yellow_image
    # 1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    list_overlapping_blue_polygon = []
    list_overlapping_yellow_polygon = []

    down_count = 0
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # for yolov3 or yolov5
    # detector = Detector()
    detector = YoloDarknet(cfg_path=Path('D:/whole/nfb_axle_cam/model/yolo.cfg'),
                           weights_path=Path('D:/whole/nfb_axle_cam/model/yolo_best.weights'),
                           class_names=['license_plate', 'axle', 'bumper', 'headlamp'],
                           darknet_dir=Path('D:/projects/darknet/Release/')
                           )

    video_path = "D:/whole/nfb_axle_cam/2023-11-30/20231115T153000.mp4"
    # capture = cv2.VideoCapture(video_path)
    capture = create_from_file_system(video_path)

    # uncomment if you need Optical Flow
    # _, prev = capture.read()
    # prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    trackers_info = []

    while True:
        # _, im = capture.read()
        im = capture.read_current()
        if im is None:
            break
        # 1920x1080->960x540
        im = cv2.resize(im, (960, 540))
        list_bboxs = []
        # bboxes = detector.detect(im)
        classes, confs, rects = detector.detect(im)
        bboxes = [(x[2][0], x[2][1], x[2][2], x[2][3], x[0], x[1]) for x in list(zip(classes, confs, rects)) if
                  x[0] == 'axle']

        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
        else:
            output_image_frame = im

        # output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            for item_bbox in list_bboxs:
                frame_id = capture.get_current_position()
                x1, y1, x2, y2, label, track_id = item_bbox

                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                y = y1_offset
                x = x1
                print(f'track_id: {track_id}, {label},  x={x}, y={y}, frame_id: {frame_id}')

                trackers_info.append({'track_id': track_id, 'class': label, "track_x": x, "track_y": y, "frame_id": frame_id})

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)

                    if track_id in list_overlapping_yellow_polygon:
                        up_count += 1
                        list_overlapping_yellow_polygon.remove(track_id)
                    else:
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)

                    if track_id in list_overlapping_blue_polygon:
                        down_count += 1
                        list_overlapping_blue_polygon.remove(track_id)
                    else:
                        pass
                else:
                    pass

            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break

                if not is_found:
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)

                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)

            list_overlapping_all.clear()
            list_bboxs.clear()
        else:
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()

        text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)

        # uncomment if you need Optical Flow
        # gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # prev_gray = gray

        key = cv2.waitKey(30) & 0xff

        if key == ord('p'):
            while True:
                key = cv2.waitKey(0) & 0xff

                if key == ord('f'):
                    curr_frame_poss = capture.get_current_position()
                    capture.set_position(curr_frame_poss + 1)
                    print(capture.get_current_position())

                if key == ord('d'):
                    curr_frame_poss = capture.get_current_position()
                    capture.set_position(curr_frame_poss - 1)
                    print(capture.get_current_position())

                if key == ord('p'):
                    break

                if key == ord('q'):
                    capture.release()
                    cv2.destroyAllWindows()
                    break

        if key == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break
        print(capture.get_current_position())
        capture.read_next()
        cv2.imshow('demo', output_image_frame)
        # uncomment if you need Optical Flow
        # cv2.imshow('flow', draw_flow(gray, flow))
        # cv2.imshow('flow HSV', draw_hsv(flow))
        # cv2.setMouseCallback('demo', onMouse)
        # cv2.waitKey(1)

    dump_trackers(trackers_info, video_path)


if __name__ == '__main__':
    main()
