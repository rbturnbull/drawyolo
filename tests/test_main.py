from typer.testing import CliRunner
from drawyolo.main import app
from unittest.mock import patch

from .test_draw import IMAGE, LABELS, MockYoloModel

runner = CliRunner()

    # image: Path=typer.Argument(..., help="Input image path"),
    # output: Path=typer.Argument(..., help="Output image path"),
    # labels: Path=typer.Option(None, help="Path to labels text file in YOLO format"),
    # weights: Path=typer.Option(None, help="Path to YOLO model weights file if not using labels text file"),
    # classes:str=typer.Option(None, help="Class names separated by comma"),
    # highest:bool=typer.Option(False, help="Only draw the box with the highest confidence"),


def test_main_labels(tmp_path):
    output = tmp_path / "output.jpg"
    result = runner.invoke(app, [
        str(IMAGE),
        str(output),
        "--labels", str(LABELS),
        "--classes", "eye,ear",
    ])
    assert result.exit_code == 0
    assert output.exists()


@patch('ultralytics.YOLO', return_value=MockYoloModel())
def test_main_model(mock_yolo, tmp_path):
    output = tmp_path / "output.jpg"
    result = runner.invoke(app, [
        str(IMAGE),
        str(output),
        "--weights", str("weights"),
    ])

    assert result.exit_code == 0
    assert output.exists()
    assert mock_yolo.called


def test_main_error(tmp_path):
    output = tmp_path / "output.jpg"
    result = runner.invoke(app, [
        str(IMAGE),
        str(output),
    ])

    assert result.exit_code == 2
    assert "You must provide either --labels or --weights." in result.stdout
    assert not output.exists()
