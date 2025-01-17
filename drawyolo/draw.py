import cv2
import random
import numpy as np
from pathlib import Path

FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_font_scale(height) -> float:
    target_height = min(max(0.03 * height, 10), 100)  # Limit to 10-100 pixels
    text = "Hg"
    low, high = 0.01, 10  # Start with a wide range
    best_fontScale = low

    while high - low > 0.01:  # Precision threshold
        mid = (low + high) / 2
        (_, text_height), _ = cv2.getTextSize(text, FONT, mid, thickness=1)

        if text_height < target_height:
            best_fontScale = mid  # Update best found value
            low = mid  # Increase font scale
        else:
            high = mid  # Decrease font scale

    return best_fontScale


def rescale_image(
    image: np.ndarray, width: int | None = None, height: int | None = None
) -> np.ndarray:
    """Rescale an image based on given width and height.

    - If both `width` and `height` are None or 0, return the original image.
    - If only one is provided, rescale while maintaining the aspect ratio.
    - If both are provided, rescale to the exact size (ignoring aspect ratio).
    """
    if not width and not height:
        return image

    original_height, original_width = image.shape[:2]
    if not width:
        scale = height / original_height
        width = int(original_width * scale)
    elif not height:
        scale = width / original_width
        height = int(original_height * scale)

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_image(
    image: np.ndarray,
    output: str | Path | None,
):
    """Save image to file"""
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)


def default_line_thickness(width: int, height: int) -> int:
    return max(round(0.002 * min(height,width)),2)


def plot_one_box(
    x, image, color=None, label: str = None, 
    line_thickness: int | None = None,
    font_scale:float|None = None,
):
    """Plots one bounding box on image img"""
    line_thickness = line_thickness or default_line_thickness(
        image.shape[1], image.shape[0]
    )
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    font_scale = font_scale or get_font_scale(image.shape[0])
    if label:
        label = str(label).upper()
        t_size = cv2.getTextSize(label, FONT, fontScale=font_scale, thickness=1)[0]
        font_thickness = max(round(t_size[1] * 0.08), 1)
        c2 = c1[0] + t_size[0], c1[1] - round(t_size[1] * 1.1) - font_thickness * 2
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (c1[0], c1[1] - round(0.05 * t_size[1]) - font_thickness),
            FONT,
            font_scale,
            [225, 255, 255],
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )


def make_colors(count: int):
    """Generate random colors"""
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(count)]
    return colors


def draw_box_on_image_with_labels(
    image: Path,
    labels: Path,
    output: Path,
    classes: list[str],
    colors: list[str] = None,
    width: int | None = None,
    height: int | None = None,
    line_thickness: int | None = None,
) -> np.ndarray:
    """
    Adds rectangle boxes on the images.
    """
    colors = colors or make_colors(len(classes))

    # Read image
    try:
        image = cv2.imread(str(image))
        image = rescale_image(image, width, height)
        height, width = image.shape[:2]        
    except Exception as e:
        print(f"Cannot read image: {e}")
        return

    line_thickness = line_thickness or default_line_thickness(width, height)
    font_scale = get_font_scale(height)

    # Get Labels
    labels = Path(labels)
    labels = labels.read_text().strip().split("\n") if labels.exists() else []
    for line in labels:
        staff = line.split()
        class_idx = int(staff[0])

        x_center, y_center, w, h = (
            float(staff[1]) * width,
            float(staff[2]) * height,
            float(staff[3]) * width,
            float(staff[4]) * height,
        )
        x1 = round(x_center - w / 2)
        y1 = round(y_center - h / 2)
        x2 = round(x_center + w / 2)
        y2 = round(y_center + h / 2)

        plot_one_box(
            [x1, y1, x2, y2],
            image,
            color=colors[class_idx],
            label=classes[class_idx],
            line_thickness=line_thickness,
            font_scale=font_scale,
        )

    save_image(image, output)
    return image


def draw_box_on_image_with_yolo_result(
    image: Path,
    results,
    output: Path,
    classes: list[str] = None,
    colors: list[str] = None,
    highest: bool = False,
    width: int | None = None,
    height: int | None = None,
    line_thickness: int | None = None,
) -> np.ndarray:
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

    original_height, original_width = image.shape[:2]
    image = rescale_image(image, width, height)
    height, width = image.shape[:2]

    font_scale = get_font_scale(height)
    line_thickness = line_thickness or default_line_thickness(width, height)

    for boxes in boxes_list:
        class_idx = boxes.cls[0].int().item()
        xyxy = boxes.xyxy.cpu()[0].clone().detach().numpy()
        xyxy[0] = xyxy[0] * width / original_width
        xyxy[1] = xyxy[1] * height / original_height
        xyxy[2] = xyxy[2] * width / original_width
        xyxy[3] = xyxy[3] * height / original_height
        plot_one_box(
            xyxy,
            image,
            color=colors[class_idx],
            label=classes[class_idx],
            line_thickness=line_thickness,
            font_scale=font_scale,
        )

    save_image(image, output)
    return image


def draw_box_on_image_with_model(
    image: Path,
    weights: Path | str,
    output: Path,
    res: int = 1280,
    classes: list[str] = None,
    colors: list[str] = None,
    highest: bool = False,
    width: int | None = None,
    height: int | None = None,
    line_thickness: int | None = None,
) -> np.ndarray:
    """Draw boxes on image using YOLO model"""
    from ultralytics import YOLO

    model = YOLO(str(weights))

    res = res or 1280
    classes = classes or model.names

    results = model.predict(source=[image], show=False, save=False, imgsz=res)
    assert len(results) == 1

    return draw_box_on_image_with_yolo_result(
        image,
        results[0],
        output=output,
        classes=classes,
        colors=colors,
        highest=highest,
        width=width,
        height=height,
        line_thickness=line_thickness,
    )
