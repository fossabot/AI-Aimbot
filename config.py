import numpy as np

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    RESOLUTION_GAME_WIDTH = 2560 
    RESOLUTION_GAME_HEIGHT = 1440 

    RESOLUTION_CAPTURE_LR_WIDTH = 200
    RESOLUTION_CAPTURE_LR_HEIGHT = 200

    RESOLUTION_CAPTURE_SR_WIDTH = 400
    RESOLUTION_CAPTURE_SR_HEIGHT = 400

    RESOLUTION_CAPTURE_WR_WIDTH = 1280
    RESOLUTION_CAPTURE_WR_HEIGHT = 720

    capture_mode =	{
    "wr": [RESOLUTION_CAPTURE_LR_WIDTH,RESOLUTION_CAPTURE_LR_HEIGHT],
    "sr": [RESOLUTION_CAPTURE_SR_WIDTH,RESOLUTION_CAPTURE_SR_HEIGHT],
    "lr": [RESOLUTION_CAPTURE_WR_WIDTH,RESOLUTION_CAPTURE_WR_HEIGHT]
    }


####

    SIS_ENTITIES = ['enemy','enemy_head','enemy_torso','enemy_arm','enemy_leg']


##############################################
    PATH_MODELS = '/models/'

    PATH_AUTOENCODER_TRAIN = '/datasets/sis/train/'

    PATH_AUTOENCODER_TEST = '/datasets/sis/test/'

##############################################

    GPU_COUNT = 1

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9



    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

if __name__ == '__main__':
    c = Config()
    # c.display()
    print(len(c.SIS_ENTITIES_SR))