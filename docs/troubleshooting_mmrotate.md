# MMRotate
## RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`
### Instance 1
`<your-virtual-env>/lib/python3.8/site-packages/mmrotate/core/post_processing/bbox_nms_rotated.py, line 59, in multiclass_nms_rotated
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`

Add `inds = inds.to(torch.device('cpu'))` to line 58

### Instance 2
`<your-virtual-env>/lib/python3.8/site-packages/mmrotate/core/post_processing/bbox_nms_rotated.py, line 88, in multiclass_nms_rotated
    labels = labels[keep]
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`

Add `keep = keep.to(torch.device('cpu'))` to line 86

## KeyError: 'RotatedRetinaNet is not in the models registry'

Add the following lines to your config file:
`custom_imports = dict(
    imports=['mmrotate.models.detectors.rotated_retinanet'],
    allow_failed_imports=False)`


## For distributed training when getting error concerning local_rank
in `<your-virtual-env>/lib/<python-version>/site-packages/torch/distributed/run.py` line 752,
change `local-rank` to `local_rank`
