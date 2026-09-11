import cv2
from src.obj_tracking_system import ObjectTrackingSystem


cap = cv2.VideoCapture(0)

tracking_system = ObjectTrackingSystem()

while True:

    success, frame = cap.read()

    if not success:
        break

    tracked = tracking_system.process_frame(
        frame
    )

    # ----------------------------------------------------
    # Draw tracked objects
    # ----------------------------------------------------

    if tracked is not None and len(tracked) > 0:

        for box, tracker_id in zip(
            tracked.xyxy,
            tracked.tracker_id
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            tracker_id = int(
                tracker_id
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {tracker_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Object Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
