import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from satellitepy.data.labels import read_label
from satellitepy.data.patch import get_patches, shift_bboxes
from satellitepy.data.utils import init_satellitepy_label


class SparseGeometryTests(unittest.TestCase):
    def test_dota_does_not_derive_hbb(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'label.txt'
            path.write_text('10 10 10 20 30 20 30 10 plane 0\n')
            labels = read_label(path, 'dota_all')
        self.assertEqual(labels['_provided_geometry'], ['obboxes'])
        self.assertEqual(len(labels['obboxes']), 1)
        self.assertIsNone(labels['hbboxes'][0])

    def test_patch_accepts_obb_without_hbb(self):
        labels = init_satellitepy_label()
        labels['_provided_geometry'] = ['obboxes']
        labels['_provided_tasks'] = []
        labels['obboxes'].append([[10, 10], [10, 20], [30, 20], [30, 10]])
        for task in labels:
            if task.startswith('_') or task == 'obboxes' or isinstance(labels[task], dict):
                continue
            labels[task].append('airplane' if task == 'coarse-class' else None)
        # Fill nested attribute lists for the same object.
        for group in labels['attributes'].values():
            for values in group.values():
                values.append(None)
        patches = get_patches(np.ones((64, 64, 3), dtype=np.uint8), labels, 0.5, 64, 0)
        result = patches['labels'][0]
        self.assertIsNone(result['hbboxes'][0])
        self.assertIsNotNone(result['obboxes'][0])
        self.assertEqual(result['_provided_geometry'], ['obboxes'])

    def test_shift_mask_accepts_potsdam_n_by_two_coordinates(self):
        labels = init_satellitepy_label()
        labels['hbboxes'].append([[10, 20], [10, 30], [20, 30], [20, 20]])
        labels['obboxes'].append(None)
        labels['masks'].append([[10, 20], [11, 21], [100, 100]])
        patch_dict = {'labels': [labels]}
        shifted = shift_bboxes(patch_dict, 0, ['hbboxes', 'obboxes'], (10, 20), 32)
        self.assertEqual(shifted['labels'][0]['masks'][0], [[0, 1], [0, 1]])


if __name__ == '__main__':
    unittest.main()
