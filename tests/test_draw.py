from pathlib import Path
import torch
import cv2
from unittest.mock import patch

from drawyolo.draw import (
    plot_one_box,
    make_colors,
    draw_box_on_image_with_labels,
    draw_box_on_image_with_model,
    draw_box_on_image_with_yolo_result,
)

TEST_DATA = Path(__file__).parent / "test-data"
IMAGE = TEST_DATA / "terrier.webp"
LABELS = TEST_DATA / "labels.txt"

def test_make_colors():
    colors = make_colors(5)
    assert len(colors) == 5
    assert all(len(color) == 3 for color in colors)
    assert all(0 <= val <= 255 for color in colors for val in color)


def test_plot_one_box():
    image = cv2.imread(str(IMAGE))
    plot_one_box([100, 100, 200, 200], image, color=[255, 0, 0], label="Test", line_thickness=2)
    assert image is not None  # Ensure the image was modified (no crash)


def test_draw_box_on_image_with_labels(tmp_path):
    output = tmp_path / "output.jpg"
    draw_box_on_image_with_labels(IMAGE, LABELS, output, ["eye", "ear"])
    assert output.exists()
    result_image = cv2.imread(str(output))
    assert result_image is not None


def test_draw_box_on_image_with_labels_width(tmp_path):
    output = tmp_path / "output.jpg"
    draw_box_on_image_with_labels(IMAGE, LABELS, output, ["eye", "ear"], width=100)
    assert output.exists()
    result_image = cv2.imread(str(output))
    height, width = result_image.shape[:2]
    assert width == 100
    assert height == 100
    assert result_image is not None


def test_draw_box_on_image_with_labels_height(tmp_path):
    output = tmp_path / "output.jpg"
    draw_box_on_image_with_labels(IMAGE, LABELS, output, ["eye", "ear"], height=140)
    assert output.exists()
    result_image = cv2.imread(str(output))
    height, width = result_image.shape[:2]
    assert width == 140
    assert height == 140
    assert result_image is not None


def test_draw_box_on_image_with_labels_width_height(tmp_path):
    output = tmp_path / "output.jpg"
    draw_box_on_image_with_labels(IMAGE, LABELS, output, ["eye", "ear"], width=100, height=140)
    assert output.exists()
    result_image = cv2.imread(str(output))
    height, width = result_image.shape[:2]
    assert width == 100
    assert height == 140
    assert result_image is not None


def test_draw_box_on_image_with_labels_fail(tmp_path):
    output = tmp_path / "output.jpg"
    draw_box_on_image_with_labels('x.jpg', LABELS, output, ["eye", "ear"])
    assert not output.exists()


class MockBoxes():
    def __init__(self, boxes):
        self.xyxy = torch.tensor([boxes[0:4]])
        self.cls = torch.tensor([boxes[-1]])
        self.conf = torch.tensor([boxes[4]])


class MockImageResult():
    def __init__(self, boxes):
        self.boxes = [MockBoxes(box) for box in boxes]


class MockYoloOutput():
    def __init__(self, image):
        self.images = [image]
        self.names = [f"category{i}" for i in range(4)]
        self.predictions = [
            MockImageResult([[300,300,440,440,0.9,0],[600,300,740,440,0.8,0]]),
        ]
        self._current_index = 0

    def __iter__(self):
        return self
    
    def __getitem__(self, index):
        return self.predictions[index]

    def __next__(self):
        if self._current_index < len(self.predictions):
            p = self.predictions[self._current_index]
            self._current_index += 1
            return p

        raise StopIteration

    def __len__(self):
        return len(self.predictions)


class MockYoloModel():
    def __init__(self):
        self.names = ["eye", "ear"]
    
    def predict(self, *args, **kwargs):
        return MockYoloOutput([IMAGE])


def test_draw_box_on_image_with_yolo_result_highest(tmp_path):
    output = tmp_path / "output_model.jpg"
    yolo_output = MockYoloOutput([IMAGE])
    result = yolo_output.predictions[0]
    draw_box_on_image_with_yolo_result(IMAGE, result, output, classes=["eye", "ear"], highest=True)
    assert output.exists()
    result_image = cv2.imread(str(output))
    assert result_image is not None


def test_draw_box_on_image_with_yolo_result(tmp_path):
    output = tmp_path / "output_model.jpg"
    yolo_output = MockYoloOutput([IMAGE])
    result = yolo_output.predictions[0]
    draw_box_on_image_with_yolo_result(IMAGE, result, output, classes=["eye", "ear"], highest=False)
    assert output.exists()
    result_image = cv2.imread(str(output))
    assert result_image is not None


@patch('ultralytics.YOLO', return_value=MockYoloModel())
def test_draw_box_on_image_with_model(mock_yolo, tmp_path):
    output = tmp_path / "output_model.jpg"
    draw_box_on_image_with_model(IMAGE, None, output, res=1280, highest=True)
    assert output.exists()
    result_image = cv2.imread(str(output))
    assert result_image is not None

    assert mock_yolo.called