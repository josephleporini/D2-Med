"""Fixed site and class order used everywhere (model heads, labels, output)."""
SITES = [("upper_extremity", "left"), ("upper_extremity", "right"),
         ("lower_extremity", "left"), ("lower_extremity", "right")]
CLASSES = ["no_injury", "wound", "amputation", "not_testable"]
# Mirroring an image flips anatomical chirality: left and right sites trade places.
LR_SWAP = [1, 0, 3, 2]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
