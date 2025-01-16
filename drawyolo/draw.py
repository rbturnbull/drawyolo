import cv2
import random
import numpy as np
from pathlib import Path


def rescale_sizes(
    original_width: int,
    original_height: int,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int]:
    """
    The size of the image after rescaling.

    - If both `width` and `height` are None or 0, return the original image.
    - If only one is provided, rescale while maintaining the aspect ratio.
    - If both are provided, rescale to the exact size (ignoring aspect ratio).
    """
    if not width and not height:
        return original_height, original_width

    if not width:
        scale = height / original_height
        width = int(original_width * scale)
    elif not height:
        scale = width / original_width
        height = int(original_height * scale)

    return width, height


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
    width, height = rescale_sizes(original_width, original_height, width, height)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_image(
    image: np.ndarray,
    output: str | Path | None,
    width: int | None = None,
    height: int | None = None,
):
    """Save image to file"""
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        image = rescale_image(image, width, height)
        cv2.imwrite(str(output), image)


def default_line_thickness(width: int, height: int) -> int:
    return round(0.002 * (height + width) / 2) + 1


def default_line_thickness_before_rescale(
    original_width: int,
    original_height: int,
    width: int | None = None,
    height: int | None = None,
) -> int:
    original_thickness_no_rescale = default_line_thickness(
        original_width, original_height
    )
    final_width, final_height = rescale_sizes(
        original_width, original_height, width, height
    )
    final_thickness = default_line_thickness(final_width, final_height)
    return round(original_thickness_no_rescale**2 / final_thickness)


def plot_one_box(
    x, image, color=None, label: str = None, line_thickness: int | None = None
):
    """Plots one bounding box on image img"""
    line_thickness = line_thickness or default_line_thickness(
        image.shape[1], image.shape[0]
    )
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[
            0
        ]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (c1[0], c1[1] - 2),
            0,
            line_thickness / 3,
            [225, 255, 255],
            thickness=tf,
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
        original_height, original_width = image.shape[:2]
    except Exception as e:
        print(f"Cannot read image: {e}")
        return

    line_thickness = line_thickness or default_line_thickness_before_rescale(
        original_width, original_height, width, height
    )

    # Get Labels
    labels = Path(labels)
    labels = labels.read_text().strip().split("\n") if labels.exists() else []
    for line in labels:
        staff = line.split()
        class_idx = int(staff[0])

        x_center, y_center, w, h = (
            float(staff[1]) * original_width,
            float(staff[2]) * original_height,
            float(staff[3]) * original_width,
            float(staff[4]) * original_height,
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
        )

    save_image(image, output, width=width, height=height)
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
    line_thickness = line_thickness or default_line_thickness_before_rescale(
        original_width, original_height, width, height
    )

    for boxes in boxes_list:
        class_idx = boxes.cls[0].int().item()
        xyxy = boxes.xyxy.cpu()[0]
        plot_one_box(
            xyxy,
            image,
            color=colors[class_idx],
            label=classes[class_idx],
            line_thickness=line_thickness,
        )

    save_image(image, output, width, height)
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
