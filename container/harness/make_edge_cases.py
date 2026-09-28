#!/usr/bin/env python3
"""Build an input directory of hostile-but-legal inputs (robustness level L4).

Each case exercises one ingest rule. Expected behavior is in EXPECT: the image must
appear in predictions.json (ICD requires a record for every input image), even if
its content can only get the fallback class. Non-image files and subdirectories
must be ignored.
"""
import io, os, random, shutil, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from make_toy_data import scene, render

EXPECT = {
    "normal.jpg": "predicted", "UPPER_CASE.JPG": "predicted", "long_ext.jpeg": "predicted",
    "gray.png": "predicted", "rgba.png": "predicted", "palette.png": "predicted", "cmyk.jpg": "predicted",
    "exif_rotated.jpg": "predicted", "huge_6000x4000.jpg": "predicted", "tiny_12x12.png": "predicted",
    "space and ünïcode.jpg": "predicted",
    "corrupt.jpg": "fallback", "truncated.jpg": "fallback", "zero_bytes.png": "fallback",
    "too_small_4x4.png": "fallback", "png_named_as.jpg": "predicted",
}
IGNORED = ["notes.txt", "labels.json", ".hidden_file", "subdir"]


def build(out: Path, n_bulk: int = 0, seed: int = 11):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = random.Random(seed)
    base = render(scene(rng, 256, 192), noise_seed=1)
    base.save(out / "normal.jpg", quality=92)
    base.save(out / "UPPER_CASE.JPG", quality=92)
    base.save(out / "long_ext.jpeg", quality=92)
    base.convert("L").save(out / "gray.png")
    base.convert("RGBA").save(out / "rgba.png")
    base.convert("P").save(out / "palette.png")
    base.convert("CMYK").save(out / "cmyk.jpg")
    ex = base.rotate(90, expand=True); exif = Image.Exif(); exif[0x0112] = 8
    ex.save(out / "exif_rotated.jpg", exif=exif)
    base.resize((6000, 4000)).save(out / "huge_6000x4000.jpg", quality=85)
    base.resize((12, 12)).save(out / "tiny_12x12.png")
    base.save(out / "space and ünïcode.jpg")
    (out / "corrupt.jpg").write_bytes(os.urandom(4096))
    buf = io.BytesIO(); base.save(buf, "JPEG"); (out / "truncated.jpg").write_bytes(buf.getvalue()[: len(buf.getvalue()) // 3])
    (out / "zero_bytes.png").write_bytes(b"")
    base.resize((4, 4)).save(out / "too_small_4x4.png")
    base.save(out / "png_named_as.jpg", format="PNG")
    (out / "notes.txt").write_text("not an image")
    (out / "labels.json").write_text("{}")
    (out / ".hidden_file").write_text("x")
    (out / "subdir").mkdir(); base.save(out / "subdir" / "nested.jpg")
    for i in range(n_bulk):
        render(scene(rng, 256, 192), noise_seed=i).save(out / f"bulk_{i:05d}.jpg", quality=90)
    return out


if __name__ == "__main__":
    build(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print("edge-case input dir written")
