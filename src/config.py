Base_Dir = 'input/images/'

class CFG():
    fold_num = 5
    seed = 48
    model_arch = 'tf_efficientnet_b7_ns'
    img_size=  512
    epochs = 10
    train_bs = 16
    valid_bs= 32
    lr= 1e-4
    min_lr= 1e-6
    weight_decay=1e-6
    num_workers= 4
    accum_iter= 2 # suppoprt to do batch accumulation for backprop with effectively larger batch size
    verbose_step= 1
    device= 'cuda:0'
    pretrained = 'imagenet'
    T_0=10
