"""Day 1 camera probe. Run from an authorized Terminal: python scripts/probe_camera.py [index]
Needs opencv-python-headless. Saves one frame to ./probe_frame.jpg and prints timing."""

from __future__ import annotations
import sys
import time

import cv2

cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0, cv2.CAP_AVFOUNDATION)
print("opened:", cap.isOpened())
start, frame = time.time(), None
for _ in range(50):
    ok, f = cap.read()
    if ok and f is not None and f.mean() > 1:  # first frames are often black while exposure settles
        frame = f
        break
    time.sleep(0.1)
cap.release()
if frame is None:
    sys.exit("no frame: grant camera access to this app in System Settings > Privacy & Security > Camera")
cv2.imwrite("probe_frame.jpg", frame)
print(f"frame {frame.shape} mean={frame.mean():.1f} in {time.time() - start:.2f}s -> probe_frame.jpg")
