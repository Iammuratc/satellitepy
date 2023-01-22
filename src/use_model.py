import torch
print(torch.cuda.is_available())

import mmcls

print(mmcls.__version__)

from mmcls.apis import init_model

config_file="/home/louis/Dropbox/Unizeug/VisualComputing/Workspace/Test/effnet_1.py"
checkpoint_file="/home/louis/Stuff/Workspace/airplane_recognition/work_dir/epoch_25.pth"

model = init_model(config_file, checkpoint_file, device='cpu')

from mmcls.apis import inference_model

result = inference_model(model, "/home/louis/Dropbox/Unizeug/VisualComputing/Workspace/Test/plane.png")



print(result)

from mmcls.apis import show_result_pyplot
show_result_pyplot(model, "/home/louis/Dropbox/Unizeug/VisualComputing/Workspace/Test/plane.png", result)



# HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 /home/louis/Dropbox/Unizeug/VisualComputing/Workspace/mmclassification/tools/test.py /home/louis/Dropbox/Unizeug/VisualComputing/Workspace/Test/effnet_1.py /home/louis/Stuff/Workspace/airplane_recognition/work_dir/epoch_25.pth --metrics accuracy