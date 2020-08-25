from .object_detector import Detector
import logging

class Model_Creator():

    def __call__(self, config):

        application2model = {
            'object_detection': Detector
        }

        self.logger = logging.getLogger("semantic_segmentation")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        if config['application'] not in application2model.keys():
            self.logger.warning(f"Model for {config['application']} do not exists. Return None.")
            return None

        return application2model[config['application']]()