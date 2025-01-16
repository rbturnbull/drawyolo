import cv2
import random
import numpy as np
from pathlib import Path


def save_image(image:np.ndarray, output:str|Path|None):
    """ Save image to file """
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)


def plot_one_box(x, image, color=None, label:str=None, line_thickness=None):
    """ Plots one bounding box on image img """

    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def make_colors(count:int):
    """ Generate random colors """
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(count)]
    return colors


def draw_box_on_image_with_labels(image:Path, labels:Path, output:Path, classes:list[str], colors:list[str]=None) -> np.ndarray:
    """
    Adds rectangle boxes on the images.
    """
    colors = colors or make_colors(len(classes))

    # Read image
    image = cv2.imread(str(image))
    try:
        height, width, _ = image.shape
    except Exception as e:
        print(f'Cannot read image: {e}')
        return

    # Get Labels
    labels = Path(labels)
    labels = labels.read_text().strip().split("\n") if labels.exists() else []
    for line in labels:
        staff = line.split()
        class_idx = int(staff[0])

        x_center, y_center, w, h = float(staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2)

        plot_one_box(
            [x1, y1, x2, y2], 
            image, 
            color=colors[class_idx],
            label=classes[class_idx], 
            line_thickness=None,
        )
    
    save_image(image, output)
    return image


def draw_box_on_image_with_yolo_result(image:Path, results, output:Path, classes:list[str]=None, colors:list[str]=None, highest:bool=False) -> np.ndarray:
    colors = colors or make_colors(len(classes))

    image = cv2.imread(str(image))

    boxes_list = results.boxes
    if highest:
        best_confidence = 0
        best_boxes = None
        for boxes in boxes_list:
            confidence = boxes.conf.item()
            if confidence > best_confidence:
                best_boxes = boxes
                best_confidence = confidence
        boxes_list = [best_boxes]

    for boxes in boxes_list:
        class_idx = boxes.cls[0].int().item()
        xyxy = boxes.xyxy.cpu()[0]
        plot_one_box(
            xyxy, 
            image, 
            color=colors[class_idx],
            label=classes[class_idx], 
            line_thickness=None,
        )

    save_image(image, output)
    return image


def draw_box_on_image_with_model(image:Path, weights:Path|str, output:Path, res:int=1280, classes:list[str]=None, colors:list[str]=None, highest:bool=False) -> np.ndarray:
    """ Draw boxes on image using YOLO model """
    from ultralytics import YOLO

    model = YOLO(str(weights))

    res = res or 1280
    classes = classes or model.names

    results = model.predict(source=[image], show=False, save=False, imgsz=res)
    assert len(results) == 1

    return draw_box_on_image_with_yolo_result(image, results[0], output, classes, colors, highest)
