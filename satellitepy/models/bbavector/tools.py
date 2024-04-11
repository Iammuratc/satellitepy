from satellitepy.models.bbavector import ctrbox_net, decoder, train_model
from satellitepy.data.utils import get_task_dict


def get_model_decoder(tasks, K, conf_thresh, target_task):
    model_decoder = decoder.DecDecoder(K=K,
        conf_thresh=conf_thresh,
        tasks=tasks, target_task=target_task)
    return model_decoder
