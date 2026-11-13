import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'tools/data/create_original_dataset_patches.py'
SPEC = importlib.util.spec_from_file_location('create_original_dataset_patches', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sample(folder, name):
    return MODULE.Sample(folder / f'{name}.png', folder / f'{name}.txt')


def test_split_is_deterministic_and_disjoint(tmp_path):
    samples = tuple(_sample(tmp_path, str(index)) for index in range(100))

    first = MODULE.split_samples(samples, seed=42)
    second = MODULE.split_samples(tuple(reversed(samples)), seed=42)

    assert {key: [s.image for s in value] for key, value in first.items()} == {
        key: [s.image for s in value] for key, value in second.items()
    }
    assert [len(first[key]) for key in ('train', 'val', 'test')] == [80, 10, 10]
    assert len({s.image for values in first.values() for s in values}) == 100


def test_vedai_rgb_and_ir_are_matched_by_source_id(tmp_path):
    root = tmp_path / 'VEDAI'
    images = root / 'Vehicules1024'
    labels = root / 'Annotations1024'
    images.mkdir(parents=True)
    labels.mkdir()
    for suffix in ('co', 'ir'):
        (images / f'00000001_{suffix}.png').touch()
    (labels / '00000001.txt').touch()

    parts = MODULE._parts(root)

    assert [part.modality for part in parts] == ['rgb', 'rgbir']
    assert parts[1].samples[0].ir.name == '00000001_ir.png'


def test_dota_v2_labels_replace_v1_labels_for_matching_images(tmp_path):
    root = tmp_path / 'DOTA'
    image_folder = root / 'train' / 'images'
    old_labels = root / 'train' / 'bounding_boxes'
    new_labels = tmp_path / 'dota_v2' / 'train' / 'train' / 'annotations' / 'version2.0'
    image_folder.mkdir(parents=True)
    old_labels.mkdir(parents=True)
    new_labels.mkdir(parents=True)
    (image_folder / 'P0001.png').touch()
    (old_labels / 'P0001.txt').touch()
    (new_labels / 'P0001.txt').touch()

    parts = MODULE._parts(root, dota_v2_root=tmp_path / 'dota_v2')

    assert len(parts) == 1
    assert parts[0].samples[0].label == new_labels / 'P0001.txt'
