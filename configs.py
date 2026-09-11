import yaml

class CONFIG:
    def __init__(self, path='configs.yml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

            self.CAMERA_FPS = data['CAMERA_FPS']
            self.DETECTION_FPS = data['DETECTION_FPS']
            self.MODEL_PATH = data['MODEL_PATH']
            self.TRACKER_CONFIG = data['TRACKER_CONFIG']
            self.CLASSES = data['CLASSES']