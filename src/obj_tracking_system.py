from src.tracker import create_tracker
from src.detector import create_model
from helper.roi import ROI
from configs import CONFIG
import supervision as sv
import numpy as np
import time

class ObjectTrackingSystem:
 
    def __init__(self, config=None):

        self.config = CONFIG() if config is None else config
 
        self.model = create_model(model_path=self.config.MODEL_PATH)
 
        # Tracker is built in tracker.py and handed back here.
        self.tracker = create_tracker()
 
        # Current tracked objects, keyed by tracker_id.
        self.tracks = {}
 
        # New ROI requests waiting to be processed.
        self.pending_rois = []
 
        self.running = True
 
        self.last_detection_time = 0.0

        self.last_frame_detections = sv.Detections.empty()



    
    def add_roi(self, roi: ROI):
        """
        Called whenever the application receives a new ROI.
 
        The application does NOT stop tracking.
        It simply queues the ROI for processing.
        """
 
        self.pending_rois.append(roi)
 
    # --------------------------------------------------------
    # Get pending ROIs
    # --------------------------------------------------------
 
    def _get_pending_rois(self):
 
        rois = self.pending_rois.copy()
        self.pending_rois.clear()
 
        return rois
 
    # --------------------------------------------------------
    # Detect objects inside ROI
    # --------------------------------------------------------
 
    def detect_roi(self, frame, roi: ROI):
 
        crop = frame[
            roi.y1:roi.y2,
            roi.x1:roi.x2
        ]
 
        if crop.size == 0:
            return sv.Detections.empty()
 
        results = self.model(
            crop,
            classes=self.config.CLASSES,
            verbose=False
        )[0]
 
        detections = sv.Detections.from_ultralytics(results)
 
        if len(detections) == 0:
            return detections
 
        # Convert ROI coordinates back to full-frame coordinates.
        detections.xyxy[:, [0, 2]] += roi.x1
        detections.xyxy[:, [1, 3]] += roi.y1
 
        return detections
 
    # --------------------------------------------------------
    # Check whether detection already belongs to a track
    # --------------------------------------------------------
 
    def is_already_tracked(self, detection_box):
 
        for track in self.tracks.values():
 
            existing_box = track["box"]
 
            iou = self.calculate_iou(
                detection_box,
                existing_box
            )
 
            if iou > 0.5:
                return True
 
        return False
 
    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------
 
    @staticmethod
    def calculate_iou(box_a, box_b):
 
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
 
        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
 
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)
 
        intersection_w = max(
            0,
            intersection_x2 - intersection_x1
        )
 
        intersection_h = max(
            0,
            intersection_y2 - intersection_y1
        )
 
        intersection_area = (
            intersection_w * intersection_h
        )
 
        area_a = (
            (ax2 - ax1) *
            (ay2 - ay1)
        )
 
        area_b = (
            (bx2 - bx1) *
            (by2 - by1)
        )
 
        union_area = area_a + area_b - intersection_area
 
        if union_area == 0:
            return 0.0
 
        return intersection_area / union_area
 
    # --------------------------------------------------------
    # Sync self.tracks from the latest tracker output
    # --------------------------------------------------------
 
    def _sync_tracks(self, tracked: sv.Detections):
 
        self.tracks = {}
 
        if tracked is None or len(tracked) == 0:
            return
 
        for box, tracker_id in zip(
            tracked.xyxy,
            tracked.tracker_id
        ):
            self.tracks[int(tracker_id)] = {
                "box": box.tolist()
            }
 
    # --------------------------------------------------------
    # Process a new ROI
    # --------------------------------------------------------
 
    def process_roi(self, frame, roi):
 
        print(
            f"[ROI] "
            f"({roi.x1}, {roi.y1}) -> "
            f"({roi.x2}, {roi.y2})"
        )
 
        detections = self.detect_roi(
            frame,
            roi
        )
 
        print(
            f"[ROI] Detected "
            f"{len(detections)} objects"
        )
 
        if len(detections) == 0:
            return sv.Detections.empty()
 
        keep_mask = []
 
        for box in detections.xyxy:
 
            box = box.tolist()
 
            if self.is_already_tracked(box):
                print("[ROI] Existing object ignored")
                keep_mask.append(False)
            else:
                print("[ROI] New object found")
                keep_mask.append(True)
 
        keep_mask = np.array(keep_mask, dtype=bool)
 
        return detections[keep_mask]
 
    # --------------------------------------------------------
    # Run YOLO on full frame
    # --------------------------------------------------------
 
    def detect_full_frame(self, frame):
 
        results = self.model(
            frame,
            classes=self.config.CLASSES,
            verbose=False
        )[0]
 
        return sv.Detections.from_ultralytics(results)
 
    # --------------------------------------------------------
    # Update tracker
    # --------------------------------------------------------
 
    def update_tracker(self, detections):
 
        tracked = self.tracker.update_with_detections(
            detections
        )
 
        self._sync_tracks(tracked)
 
        return tracked
 
    # --------------------------------------------------------
    # Main processing
    # --------------------------------------------------------
 
    def process_frame(self, frame):
 
        current_time = time.monotonic()
 
        # ====================================================
        # STATE 1:
        # Process new ROIs
        #
        # This can happen at ANY time.
        # Tracking does not stop.
        # ====================================================
 
        new_roi_detections = sv.Detections.empty()
 
        pending_rois = self._get_pending_rois()
 
        for roi in pending_rois:
 
            roi_detections = self.process_roi(
                frame,
                roi
            )
 
            if len(roi_detections) > 0:
                new_roi_detections = sv.Detections.merge(
                    [new_roi_detections, roi_detections]
                )
 
        # ====================================================
        # STATE 2:
        # YOLO detection at lower FPS
        # ====================================================
 
        detection_interval = 1.0 / self.config.DETECTION_FPS
 
        if (
            current_time - self.last_detection_time
            >= detection_interval
        ):
 
            frame_detections = self.detect_full_frame(
                frame
            )
 
            self.last_detection_time = current_time
 
        else:
 
            frame_detections = self.last_frame_detections
 
        # Merge any newly-found ROI objects into this frame's
        # detections so the tracker can pick them up immediately,
        # instead of waiting for a full-frame YOLO pass to
        # rediscover them.
        if len(new_roi_detections) > 0:
            frame_detections = sv.Detections.merge(
                [frame_detections, new_roi_detections]
            )
 
        # Remember this frame's detections so the next frame(s) can
        # carry them forward if no fresh detection arrives before
        # the next tick.
        self.last_frame_detections = frame_detections

        # ====================================================
        # STATE 3:
        # Update tracker - now called every frame, not just on
        # detection ticks.
        # ====================================================
 
        tracked = self.update_tracker(
            frame_detections
        )
 
        return tracked
