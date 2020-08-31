from .object_detection import Object_Detection
from .semantic_segmentation import Semantic_Segmentation

class Application_Creator():

    def __call__(self, server):

        application_dict = {
            'object_detection': Object_Detection,
            'semantic_segmentation': Semantic_Segmentation
        }

        return application_dict[server.config['application']](server)