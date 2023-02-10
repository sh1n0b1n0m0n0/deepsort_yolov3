import numpy as np
from pathlib import Path
import tracker
from detector import Detector
from inexlib.object_detection.yolo_detector import YoloDarknet
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def count_elems(lst):
    lst.sort()
    return {f'id_{elem}': lst.count(elem) for elem in lst}


def logging():
    pass


if __name__ == '__main__':

    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

    # 初始化2个撞线polygon
    list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                     [299, 375], [267, 289]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                       [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    detector = Detector()
    nash_detector = YoloDarknet(cfg_path=Path("D:/whole/models/yolov3_tiny_18cls_21/yolo.cfg"),
                                weights_path=Path("D:/whole/models/yolov3_tiny_18cls_21/yolo.weights"),
                                class_names='truck_cabin tank_trailer car_carrier car_trailer car_carrier_trailer trailer platform_trailer car motorcycle bus truck pickup axle unknown van dump_truck_body service_vehicle drawbar'.split(),
                                darknet_dir=Path('C:/darknet-AlexyAB/build/darknet/x64/')

    )
    # 打开视频
    video_path = "D:/projects/rv_data/data/nfb/front/20220208/part1/out_sample.mp4"
    capture = cv2.VideoCapture(video_path)
    log_file = Path(video_path).parent / 'log.txt'

    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')
    list_of_track_id = []
    with open(log_file, "w") as f:
        while True:
            # 读取每帧图片
            _, im = capture.read()
            if im is None:
                break

            # 缩小尺寸，1920x1080->960x540
            im = cv2.resize(im, (960, 540))

            list_bboxs = []
            # bboxes = detector.detect(im)
            classes, confs, rects = nash_detector.detect(im)
            bboxes = [(x[2][0], x[2][1], x[2][2], x[2][3], x[0], x[1]) for x in list(zip(classes, confs, rects)) if x[0] == 'truck']
            # 如果画面中 有bbox
            if len(bboxes) > 0:
                list_bboxs = tracker.update(bboxes, im)

                # 画框
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
                pass
            else:
                # 如果画面中 没有bbox
                output_image_frame = im
            pass

            # 输出图片
            output_image_frame = cv2.add(output_image_frame, color_polygons_image)

            if len(list_bboxs) > 0:
                # ----------------------判断撞线----------------------
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, label, track_id = item_bbox

                    # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    y1_offset = int(y1 + ((y2 - y1) * 0.6))

                    # 撞线的点
                    y = y1_offset
                    x = x1
                    print(f'track_id: {track_id}, {label}')

                    list_of_track_id.append(track_id)
                    message = [str(f'previous track_ids: {list_of_track_id}'),
                               str(f'first track: id_{list_of_track_id[0]}, {count_elems(list_of_track_id)}')]
                    if polygon_mask_blue_and_yellow[y, x] == 1:
                        # 如果撞 蓝polygon
                        if track_id not in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.append(track_id)
                        pass

                        # 判断 黄polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 外出方向
                        if track_id in list_overlapping_yellow_polygon:
                            # 外出+1
                            up_count += 1
                            f.write('\n'.join(message))
                            f.write('\n')
                            print(message[0])
                            print(message[1])
                            # print(f'previous track_ids: {list_of_track_id}')
                            # print(f'first track: id_{list_of_track_id[0]}, {count_elems(list_of_track_id)}')
                            list_of_track_id.clear()
                            # print(f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')
                            print(f'Category: {label} | id: {track_id} | Upstream | Total upstream: {up_count} | List of upstream id: {list_overlapping_yellow_polygon}')
                            # 删除 黄polygon list 中的此id
                            list_overlapping_yellow_polygon.remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass

                    elif polygon_mask_blue_and_yellow[y, x] == 2:
                        # 如果撞 黄polygon
                        if track_id not in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.append(track_id)
                        pass

                        # 判断 蓝polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 进入方向
                        if track_id in list_overlapping_blue_polygon:
                            # 进入+1
                            down_count += 1
                            f.write('\n'.join(message))
                            f.write('\n')
                            print(message[0])
                            print(message[1])
                            # print(f'previous track_ids: {list_of_track_id}')
                            # print(f'first track: {list_of_track_id[0]}, {count_elems(list_of_track_id)}')
                            list_of_track_id.clear()
                            # print(f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')
                            print(f'Category: {label} | id: {track_id} | Downstream | Total Downstream: {down_count} | List of downstream id: {list_overlapping_blue_polygon}')

                            # 删除 蓝polygon list 中的此id
                            list_overlapping_blue_polygon.remove(track_id)

                            pass
                        else:
                            pass
                        pass
                    else:
                        pass
                    pass

                pass

                # ----------------------清除无用id----------------------
                list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                for id1 in list_overlapping_all:
                    is_found = False
                    for _, _, _, _, _, bbox_id in list_bboxs:
                        if bbox_id == id1:
                            is_found = True
                            break
                        pass
                    pass

                    if not is_found:
                        # 如果没找到，删除id
                        if id1 in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.remove(id1)
                        pass
                        if id1 in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.remove(id1)
                        pass
                    pass
                list_overlapping_all.clear()
                pass

                # 清空list
                list_bboxs.clear()

                pass
            else:
                # 如果图像中没有任何的bbox，则清空list
                list_overlapping_blue_polygon.clear()
                list_overlapping_yellow_polygon.clear()
                pass
            pass

            text_draw = 'DOWN: ' + str(down_count) + \
                        ' , UP: ' + str(up_count)
            output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                             org=draw_text_postion,
                                             fontFace=font_draw_number,
                                             fontScale=1, color=(255, 255, 255), thickness=2)

            cv2.imshow('demo', output_image_frame)
            # cv2.waitKey(1)

            key = cv2.waitKey(30) & 0xff
            if key == ord('p'):
                print("Pause button pressed")
                while True:
                    key = cv2.waitKey(30) & 0xff
                    if key == ord('r'):
                        print("Resume button pressed")
                        break

            elif key == ord('q'):
                break

            pass
        pass

        # print(f'previous track_ids: {list_of_track_id}')
        # print(f'first track: {list_of_track_id[0]}, {count_elems(list_of_track_id)}')
        f.write('\n'.join(message))
        f.write('\n')
        print(message[0])
        print(message[1])
        list_of_track_id.clear()

        capture.release()
        cv2.destroyAllWindows()
