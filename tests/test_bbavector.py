from pathlib import Path

import cv2
import numpy as np
import torch

from satellitepy.dataset.bbavector.utils import Utils
from satellitepy.models.bbavector.ctrbox_net import CTRBOX
from satellitepy.models.bbavector.loss import LossAll
from satellitepy.models.bbavector.utils import load_checkpoint, save_model


def _classification_annotation(box_key):
    return {
        box_key: np.asarray([[4.0, 4.0, 4.0, 2.0]], dtype=np.float32),
        'cls_coarse-class': np.asarray([0]),
    }


def test_horizontal_box_ground_truth_does_not_require_oriented_boxes():
    utils = Utils(['hbboxes', 'coarse-class'], input_h=32, input_w=32, down_ratio=4, K=8)
    result = utils.generate_ground_truth(
        np.zeros((32, 32, 3), dtype=np.uint8),
        _classification_annotation('hbboxes'),
        'coarse-class',
    )

    assert result['hbboxes_params'].shape == (8, 2)
    assert result['reg_mask'].sum().item() == 1


def test_non_augmented_data_is_resized_with_boxes_and_mask():
    utils = Utils(['hbboxes', 'masks', 'coarse-class'], input_h=20, input_w=40, down_ratio=4, K=8)
    mask = np.zeros((10, 20), dtype=np.float32)
    mask[2:6, 3:8] = 1
    annotation = {
        'hbboxes': np.asarray([[[2.0, 2.0], [2.0, 6.0], [8.0, 6.0], [8.0, 2.0]]]),
        'masks': mask,
        'cls_coarse-class': np.asarray([0]),
    }

    image, transformed = utils.data_transform(
        np.zeros((10, 20, 3), dtype=np.uint8), annotation, augmentation=False)

    assert image.shape == (20, 40, 3)
    assert transformed['masks'].shape == (1, 20, 40)
    np.testing.assert_allclose(transformed['hbboxes'][0, :2], [10.0 / 4, 8.0 / 4])


def test_mask_head_preserves_input_resolution():
    model = CTRBOX(
        heads={'masks': 1, 'cls_coarse-class': 4, 'hbboxes_params': 2, 'hbboxes_offset': 2},
        pretrained=False,
        down_ratio=4,
        final_kernel=1,
        head_conv=16,
        resnet_type='34',
    ).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))

    assert output['masks'].shape == (1, 1, 64, 64)
    assert output['cls_coarse-class'].shape[-2:] == (16, 16)


def test_multitask_loss_backward_pass():
    tasks = ['hbboxes', 'coarse-class', 'role']
    utils = Utils(tasks, input_h=32, input_w=32, down_ratio=4, K=8)
    annotation = _classification_annotation('hbboxes')
    annotation['cls_role'] = np.asarray([0])
    ground_truth = utils.generate_ground_truth(
        np.zeros((32, 32, 3), dtype=np.uint8), annotation, 'coarse-class')
    batch = {key: value.unsqueeze(0) for key, value in ground_truth.items()}
    predictions = {
        'cls_coarse-class': torch.full((1, 4, 8, 8), 0.5, requires_grad=True),
        'cls_role': torch.full((1, 11, 8, 8), 0.5, requires_grad=True),
        'hbboxes_params': torch.zeros((1, 2, 8, 8), requires_grad=True),
        'hbboxes_offset': torch.zeros((1, 2, 8, 8), requires_grad=True),
    }

    losses = LossAll(tasks, 'coarse-class')(predictions, batch, 'coarse-class')
    sum(losses.values()).backward()

    assert all(torch.isfinite(value) for value in losses.values())
    assert predictions['cls_role'].grad is not None


def test_checkpoint_round_trip_without_validation_loss(tmp_path: Path):
    model = CTRBOX(
        heads={'cls_coarse-class': 4, 'hbboxes_params': 2, 'hbboxes_offset': 2},
        pretrained=False,
        down_ratio=4,
        final_kernel=1,
        head_conv=256,
        resnet_type='101',
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / 'model.pth'
    save_model(checkpoint, 2, model, optimizer)

    loaded_model, _, epoch, loss = load_checkpoint(checkpoint, down_ratio=4)

    assert isinstance(loaded_model, CTRBOX)
    assert epoch == 2
    assert loss is None
