import random
import cv2
import numpy as np
import os
import torch

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = cv2.cvtColor(im_bgr,cv2.COLOR_BGR2RGB)
    return im_rgb

def seed_everything(seed):
    '''
        creates seed for the environment
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True