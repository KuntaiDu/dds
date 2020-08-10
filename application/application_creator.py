from .object_detection import Object_Detection

class Application_Creator():
    def create_object_detection(self, config):
        return Object_Detection(config)

    def create_semantic_segmentation():
        print("semantic segmentation under construction")
        return None