from pathlib import Path
from typing import List
from rich.console import Console
console = Console()
import typer

from .draw import draw_box_on_image

app = typer.Typer()


@app.command()
def drawyolo(image: Path, labels: Path, output: Path, classes:str):
    """
    Draw boxes on image
    """
    classes = classes.split(',')

    draw_box_on_image(image, labels, output, classes)
