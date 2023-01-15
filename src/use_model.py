import mmcls

print(mmcls.__version__)

from mmcls.apis import init_model

config_file="../../mmclassification/configs/resnet/resnet50_8xb32_in1k.py"
checkpoint_file="../../notebooks/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth"

model = init_model(config_file, checkpoint_file, device='cpu')

from mmcls.apis import inference_model

result = inference_model(model, "../../notebooks/resources/Banana-Single.png")

print(result)

from mmcls.apis import show_result_pyplot
show_result_pyplot(model, "../../notebooks/resources/Banana-Single.png", result)