"""M1 Ingest: list the flat input directory and decode images robustly.

Decode rules (each is a derived requirement from the conformance tests):
- extensions .jpg .jpeg .png, case-insensitive; everything else is ignored and logged
- subdirectories are ignored (ICD: flat directory)
- EXIF orientation is applied (rotation never changes chirality, so labels are unaffected)
- grayscale, palette, RGBA, CMYK -> RGB
- very large JPEGs are decoded at reduced size via draft mode
- any decode failure returns None; the executive then uses the fallback for that image
"""
from pathlib import Path
from PIL import Image, ImageOps, ImageFile
from .sites import IMAGE_EXT

Image.MAX_IMAGE_PIXELS = 250_000_000     # allow large photos, still refuse absurd ones
ImageFile.LOAD_TRUNCATED_IMAGES = False  # a truncated file is a decode failure, not a guess


def list_images(input_dir: Path):
    images, ignored = [], []
    for p in sorted(input_dir.iterdir(), key=lambda q: q.name):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            images.append(p)
        else:
            ignored.append(p.name)
    return images, ignored


def decode(path: Path, target: int):
    """Return (RGB PIL image, flags) or (None, flags)."""
    flags = []
    try:
        with Image.open(path) as im:
            if im.format == "JPEG" and max(im.size) > 4 * target:
                im.draft("RGB", (2 * target, 2 * target))
                flags.append("draft_decode")
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                flags.append(f"mode_{im.mode}")
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
                    im = Image.alpha_composite(bg, im)
                im = im.convert("RGB")
            im.load()
            if min(im.size) < 8:
                flags.append("too_small")
                return None, flags
            return im.copy(), flags
    except Exception as ex:  # corrupt, truncated, unsupported, bomb
        flags.append(f"decode_error:{type(ex).__name__}")
        return None, flags
