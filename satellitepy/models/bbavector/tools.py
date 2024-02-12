from satellitepy.models.bbavector import ctrbox_net, decoder, train_model
from satellitepy.data.utils import get_task_dict

def get_model(tasks,down_ratio):
    heads = {#'hm': num_classes, # heatmap
             #'reg_wh': 10, # box param
             #'reg_bb_offset': 2, # offset
             #'cls_theta': 1 # orientation
             }

    for task in tasks:
        if task == 'fine-class':
            heads["cls_" + task] = 108
        if task == "obboxes":
            heads["obboxes_params"] = 10
            heads["obboxes_offset"] = 2
            heads["obboxes_theta"] = 1
        elif task == "hbboxes":
            heads["hbboxes_params"] = 2
            heads["hbboxes_offset"] = 2
        elif task == "masks":
            heads[task] = 1
        else:
            td = get_task_dict(task)
            if 'max' and 'min' in td.keys():
                heads["reg_" + task] = 1
            else:
                heads["cls_" + task] = len(set(td.values()))

    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)
    return model

def get_model_decoder(tasks, K, conf_thresh):
    model_decoder = decoder.DecDecoder(K=K,
        conf_thresh=conf_thresh,
        tasks=tasks)
    return model_decoder
