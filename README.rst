================================================================
drawyolo
================================================================

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/drawyolo
    :target: https://pypi.org/project/drawyolo/

.. |testing badge| image:: https://github.com/rbturnbull/drawyolo/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/drawyolo/actions

.. |docs badge| image:: https://github.com/rbturnbull/drawyolo/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/drawyolo
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/4824a2c398904709e901d0b7e8269d4b/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/drawyolo/coverage/
    
.. end-badges

.. start-quickstart

Draws boxes on images from annotations in YOLO format.

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install drawyolo

Or directly from the repository:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/drawyolo.git

Drawing boxes from labels
==================================

If you have a label file in YOLO format like this:

.. code-block:: text

    class_id x_center y_center width height

For example:

.. code-block:: text

    0 0.35 0.35 0.14 0.14
    0 0.55 0.3 0.14 0.14
    1 0.7 0.15 0.2 0.2
    
Then you can draw the boxes on an image like this:

.. code-block:: bash

    drawyolo image.jpg output.jpg --labels labels.txt --classes class1,class2

To try it out on the image in the repository:

.. code-block:: bash

    drawyolo tests/test-data/terrier.webp tests/test-data/output.jpg --labels tests/test-data/labels.txt --classes eye,ear

That will create an image like this:

.. image:: https://raw.githubusercontent.com/rbturnbull/drawyolo/main/tests/test-data/output.jpg
    :alt: Output image
    :align: center

To resize the image, use the ``--width`` and/or ``--height`` options. 
The aspect ratio will be preserved if you do not set both ``--width`` and ``--height``.

For example:

.. code-block:: bash

    drawyolo tests/test-data/terrier.webp tests/test-data/output-thumbnail.jpg --labels tests/test-data/labels.txt --classes eye,ear --width 240

.. image:: https://raw.githubusercontent.com/rbturnbull/drawyolo/main/tests/test-data/output-thumbnail.jpg
    :alt: Output image thumbnail
    :align: center

The thickness of the line will be set according to the final size of the image. You can change the thickness with the ``--line-thickness`` option.

Drawing boxes from a YOLO model
==================================

If you have a YOLO model with weights, you can draw the boxes on an image like this:

.. code-block:: bash

    drawyolo image.jpg output.jpg --weights model.pt

You can also resize the image as before with the ``--width`` and ``--height`` options.

Advanced usage
==================================

For more options see the help

.. code-block:: bash

    drawyolo --help


.. end-quickstart


Credits
==================================

.. start-credits

Robert Turnbull (Melbourne Data Analytics Platform) - `https://robturnbull.com <https://robturnbull.com>`_

.. end-credits

