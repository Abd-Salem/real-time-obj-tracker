import supervision as sv
 
 
def create_tracker() -> sv.ByteTrack:
    """
    Builds and returns a tracker instance.
 
    Kept in its own module so ObjectTrackingSystem doesn't need to
    know which tracking library/algorithm is behind it - swapping
    ByteTrack for something else later only touches this file.
    """
 
    return sv.ByteTrack()
