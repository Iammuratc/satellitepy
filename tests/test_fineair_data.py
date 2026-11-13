import json

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from satellitepy.data.labels import read_fineair_label
from satellitepy.data.utils import read_fineair_rgb_ir


def test_fineair_reader_preserves_rgb_and_separates_nir(tmp_path):
    path = tmp_path / 'scene.tif'
    bands = np.stack([
        np.full((4, 5), 100, dtype=np.uint16),  # red
        np.full((4, 5), 200, dtype=np.uint16),  # green
        np.full((4, 5), 300, dtype=np.uint16),  # blue
        np.full((4, 5), 400, dtype=np.uint16),  # NIR
    ])
    with rasterio.open(path, 'w', driver='GTiff', width=5, height=4, count=4,
                       dtype='uint16', transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(bands)

    image = read_fineair_rgb_ir(
        path, stretches=((0, 400),) * 4, output_size=(6, 7))

    assert image.shape == (6, 7, 4)
    # OpenCV on-disk order is B, G, R, NIR.
    assert image[0, 0].tolist() == [191, 127, 63, 255]
    assert not image[5, 6].any()
    output = tmp_path / 'scene.png'
    assert cv2.imwrite(str(output), image)
    assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED)[0, 0].tolist() == [191, 127, 63, 255]


def test_fineair_class_is_shared_fine_class_and_subtype_is_optional(tmp_path):
    path = tmp_path / 'labels.geojson'
    feature = {
        'properties': {
            'Type': 'B38M', 'Subtype': 'B737--MAX8',
            'fineairtype': 'B737', 'Role': 'Airliner', 'Source': 'FR24',
        },
        'geometry': {'coordinates': [[0, 0], [0, 10], [20, 10], [20, 0]]},
    }
    path.write_text(json.dumps({'features': [feature]}))

    labels = read_fineair_label(path)

    assert labels['fine-class'] == ['B737']
    assert labels['fineair-class'] == ['B737']
    assert labels['very-fine-class'] == ['B737--MAX8']
