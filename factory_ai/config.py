"""Shared constants: classes, nominal equipment sizes, image geometry."""

MACHINES = ["Cutting", "Drilling", "Assembly", "Inspection", "Packaging", "Storage"]
DOCKS = ["Receiving", "Dispatch"]          # fixed material entry / exit points
OBSTACLE = "Column"                        # fixed obstacle (pillar)
CLASSES = MACHINES + DOCKS + [OBSTACLE]    # detector foreground classes
CLASS_TO_ID = {c: i + 1 for i, c in enumerate(CLASSES)}   # 0 = background
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
NUM_DET_CLASSES = len(CLASSES) + 1

# Nominal footprints (w, h) in metres, taken from the Phase-1 notebook.
NOMINAL_SIZE = {
    "Cutting": (4.0, 3.0), "Drilling": (3.0, 3.0), "Assembly": (5.0, 3.0),
    "Inspection": (3.0, 2.5), "Packaging": (4.0, 3.0), "Storage": (4.0, 4.0),
}
DOCK_SIZE = (1.5, 3.0)   # depth into floor x length along wall (metres) for vertical wall docks
COLUMN_SIZE = 1.5

# Synthetic plan-image geometry (pixels)
IMG_W, IMG_H = 320, 240
MARGIN_L = MARGIN_R = 12
MARGIN_T = 10
MARGIN_B = 26
WALL_PX = 3
SCALE_BAR_M = 5.0
SCALE_BAR_TICK_PX = 2

# Display colours (hex) per class, used by plots
COLORS = {
    "Cutting": "#4C78A8", "Drilling": "#F58518", "Assembly": "#54A24B",
    "Inspection": "#E45756", "Packaging": "#B279A2", "Storage": "#72B7B2",
    "Receiving": "#2E7D32", "Dispatch": "#C62828", "Column": "#555555",
}
