from pathlib import Path
import typer

from .draw import draw_box_on_image_with_labels, draw_box_on_image_with_model

app = typer.Typer()

@app.command()
def drawyolo(
    image: Path, 
    output: Path,
    labels: Path=None, 
    weights: Path=None, 
    classes:str=None,
    highest:bool=False,
):
    """
    Draw boxes on image
    """
    if classes is not None:
        classes = classes.split(',')

    if labels is not None:
        draw_box_on_image_with_labels(image, labels, output, classes)
    elif weights is not None:
        draw_box_on_image_with_model(image, weights, output, classes=classes, highest=highest)
    else:
        print("Please provide either labels or a model")