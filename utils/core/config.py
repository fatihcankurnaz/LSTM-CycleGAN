from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import yaml

from easydict import EasyDict

config = EasyDict()
config.NUM_GPUS = 1
config.OUTPUT_DIR= ""
config.MODEL= "lstm_cycle_gan"

config.DATALOADER = EasyDict()
config.DATALOADER.WORKERS = 2
config.DATALOADER.SHUFFLE = True
config.DATALOADER.FIRST_ROBOT_PATH = "/home/fatih/LidarLabelsCameraViewTest"
config.DATALOADER.SECOND_ROBOT_PATH = "/home/fatih/SegmentedInputTest"


config.GENERATOR = EasyDict()
# Base Learning rate for optimizer
config.GENERATOR.BASE_LR = 0.0006
# Change learning rate in each step_size number of iterations by multiplying it with gamma
config.GENERATOR.STEP_SIZE = 5
config.GENERATOR.STEP_GAMMA = 0.1


config.DISCRIMINATOR = EasyDict()
config.DISCRIMINATOR.BASE_LR = 0.0001
config.DISCRIMINATOR.STEP_SIZE = 5
config.DISCRIMINATOR.STEP_GAMMA = 0.1

config.TRAIN = EasyDict()
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.START_EPOCH = 0
config.TRAIN.MAX_EPOCH = 1000
config.TRAIN.LOAD_WEIGHTS = ""
config.TRAIN.SAVE_WEIGHTS = ""
config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION = "mean"
config.TRAIN.CYCLE_LOSS_REDUCTION = "mean"
config.TRAIN.EXAMPLE_SAVE_PATH = ""
config.TRAIN.GRAPH_SAVE_PATH = ""
config.TRAIN.SAVE_AT = 2
config.TRAIN.CYCLE_LAMBDA = 10
config.TRAIN.TARGET_LENGTH = 10


config.TEST = EasyDict()

config.TEST.RESULT_SAVE_PATH = "/home/fatih/my_git/outputs/eval_result"
config.TEST.INPUT_DIR = "/home/fatih/Inputs/TEST"


def fix_the_type(desired_type, given_type):
    if type(desired_type) == type(given_type):
        return given_type
    elif isinstance(desired_type, bool):
        if(given_type == True):
            return True
        else:
            return False

    elif isinstance(desired_type, int):
        return int(given_type)
    elif isinstance(desired_type, float):
        return float(given_type)


def update_config_secondaries(left, right):
    for i,k in right.items():
        new_k = fix_the_type(config[left][i],k)
        config[left][i] = new_k


def load_config(config_file):
    if config_file is not None:
        with open(config_file,"r") as f:
            my_config = EasyDict(yaml.load(f,  Loader=yaml.BaseLoader) )

            for i, k in my_config.items():

                if i in config:
                    if isinstance(k, EasyDict):
                        update_config_secondaries(i,k)
                    else:
                        new_k = fix_the_type(config[i],k)
                        config[i] = new_k
                else:
                    raise ValueError(i, " is not one of the core variables")


