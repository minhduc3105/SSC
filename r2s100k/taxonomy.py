"""R2S100K label IDs and RGB palette used by the SSC data adapter."""
CLASSES = {
    1: ("Wet Surface", "Slightly watered road surface; it can be damp due to snow or cold weather.", (2, 79, 59)),
    2: ("Asphalt", "Road pavement constructed using aggregates such as crushed rocks, sand, and coal tar.", (17, 163, 74)),
    3: ("Boggy", "Unpaved road surface filled with mud.", (112, 84, 62)),
    4: ("Earthen", "Unpaved road with compacted layers of stabilized soil.", (225, 148, 79)),
    5: ("Crag-stone", "Hilltop stones dropped over the road surface in mountainous areas.", (120, 114, 104)),
    6: ("Road Grime", "Dirt ingrained on the road surface.", (166, 130, 95)),
    7: ("Vegetation Misc", "Naturally occurring vegetation other than trees adjacent to the road.", (128, 222, 91)),
    8: ("Distress", "Longitudinal and transverse cracks on the road caused by lack of maintenance.", (119, 61, 128)),
    9: ("Drainage Grate", "An elongated perforated cover or grating used to cover a water drain.", (93, 86, 176)),
    10: ("Water Puddle", "A small pool of water over the road.", (140, 160, 222)),
    11: ("Speed Breaker", "Concrete speed bumps, speed humps, and speed cushions over the road.", (234, 133, 5)),
    12: ("Misc", "An unclear object dropped over the road.", (156, 28, 39)),
    13: ("Gravel", "Unpaved road surface with loose aggregates of variable-sized rock fragments.", (99, 122, 130)),
    14: ("Concrete", "Road material made with binders such as rough and fine aggregates.", (123, 43, 31)),
}
BACKGROUND_RGB = (0, 0, 0)
NUM_CLASSES = 15
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
