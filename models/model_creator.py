from .object_detector import Detector

class Model_Creator():

    def __call__(self, config):

        application_dict = {
            'object_detection': Detector,
            'semantic_segmentation': None
        }

        return application_dict[config['application']]()