
from torchsummaryX import summary
import torch


def get_yolo(print_summary=False):
	### Model
	model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

	if print_summary:
		summary(model,torch.rand((1, 3, 512, 512)))
	return model