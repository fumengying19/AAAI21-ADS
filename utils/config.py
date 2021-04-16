''' Configuration File.
'''


class DATASETS(object):
    def __init__(self):
        self.NUM_TRAIN = 50000
        self.NUM_VAL = 50000 - self.NUM_TRAIN
        self.ROOT = {
            'cifar10': '~/datasets',
            'cifar100': '~/datasets'
        }


class ACTIVE_LEARNING(object):
    def __init__(self):
        self.TRIALS = 10
        self.CYCLES = 1
        self.ADDENDUM = 1000
        self.SUBSET = 10000


class TRAIN(object):
    def __init__(self):
        self.BATCH = 128
        self.EPOCH = 200
        self.LR = 0.1
        self.MILESTONES = [160]
        self.EPOCHL = 120
        self.MOMENTUM = 0.9
        self.WDECAY = 5e-4
        self.MIN_CLBR = 0.1
        self.MAX_CLBR = 0.1


class LEARNING_LOSS(object):
    def __init__(self):
        self.MARGIN = 1.0
        self.WEIGHT = 1.0


class global_vars(object):
    def __init__(self):
        self.iter = 0

    def update_vars(self):
        self.iter += 1


class CONFIG(object):
    def __init__(self, port=9000):
        self.port = port
        self.DATASET = DATASETS()
        self.ACTIVE_LEARNING = ACTIVE_LEARNING()
        self.TRAIN = TRAIN()
        self.LEARNING_LOSS = LEARNING_LOSS()
        self.global_iter = global_vars()


def get_configs(port=9000):
    cfg = CONFIG(port=9000)
    return cfg

