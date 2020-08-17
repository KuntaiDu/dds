from .object_detection import Object_Detection

class Application_Creator():

    def __call__(self, server):

        application_dict = {
            'object_detection': Object_Detection,
            'semantic_segmentation': None
        }

        return application_dict[server.config['application']](server)