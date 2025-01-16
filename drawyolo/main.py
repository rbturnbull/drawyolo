from pathlib import Path
import typer

from .draw import draw_box_on_image_with_labels, draw_box_on_image_with_model

app = typer.Typer()

@app.command()
def drawyolo(
    image: Path=typer.Argument(..., help="Input image path"),
    output: Path=typer.Argument(..., help="Output image path"),
    labels: Path=typer.Option(None, help="Path to labels text file in YOLO format"),
    weights: Path=typer.Option(None, help="Path to YOLO model weights file if not using labels text file"),
    classes:str=typer.Option(None, help="Class names separated by comma"),
    highest:bool=typer.Option(False, help="Only draw the box with the highest confidence"),
):
    """
    Draw boxes on image from a YOLO model or labels text file in YOLO format.
    """
    if classes is not None:
        classes = classes.split(',')

    if labels is not None:
        draw_box_on_image_with_labels(image, labels, output, classes)
    elif weights is not None:
        draw_box_on_image_with_model(image, weights, output, classes=classes, highest=highest)
    else:
        raise typer.BadParameter("You must provide either --labels or --weights.")
