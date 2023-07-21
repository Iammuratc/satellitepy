from satellitepy.models.bbavector import ctrbox_net, decoder, train_model
from satellitepy.data.utils import get_task_dict

def get_model(task,down_ratio):
    task_dict = get_task_dict(task)
    num_classes = len(task_dict)

    heads = {'hm': num_classes,
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)
    return model

def get_model_decoder(task,
    K,
    conf_thresh):

    task_dict = get_task_dict(task)
    num_classes = len(task_dict)

    model_decoder = decoder.DecDecoder(K=K,
        conf_thresh=conf_thresh,
        num_classes=num_classes)
    return model_decoder