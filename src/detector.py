from ultralytics import YOLO
 
 
def create_model(model_path: str) -> YOLO:
    """
    Builds and returns a YOLO model instance.
 
    Kept in its own module so ObjectTrackingSystem doesn't need to
    know which detector/weights are behind it - swapping yolov8n
    for another model or version later only touches this file.
    """
 
    return YOLO(model_path)
 
