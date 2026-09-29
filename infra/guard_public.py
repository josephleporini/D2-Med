"""Publish guard G3 (Adaptation Kit v1.0 §8): nothing derived from real or DARPA data may be pushed to the public
repositories (D2-Med, DDData). A pod job calls this before any commit to DDData; exit 0 means every input image is
synthetic, any other exit means do not publish.

An input image counts as synthetic only if a sidecar for it (same stem, `<stem>_sidecar.json`) exists in one of the
given directories and its provenance names our generator (`scene<N>/<version>`). Anything else refuses the publish:
an image with no sidecar, a sidecar without generator provenance, an unreadable sidecar, or an empty input set.
D2_REAL_DATA=1 in the environment refuses unconditionally (the flag a real-data run must set).

Usage: python infra/guard_public.py <image_dir> [<sidecar_dir> ...]
       python infra/guard_public.py --list <file_with_image_paths> [<sidecar_dir> ...]
"""
import os, re, sys, json, glob

GEN = re.compile(r'^scene\d+/\d')
SIDE = ('_part3', '_id', '_amodal', '_occ', '_injury')
IMG = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def check(images, sidecar_dirs):
    if os.environ.get('D2_REAL_DATA') == '1':
        return ['D2_REAL_DATA=1: real-data run, publishing to a public repository is refused']
    if not images:
        return ['no input images found: refusing (cannot show the inputs are synthetic)']
    bad = []
    for p in images:
        stem = os.path.splitext(os.path.basename(p))[0]
        for suffix in SIDE:   # generator side outputs share the scene stem
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        sc = None
        for d in [os.path.dirname(p)] + list(sidecar_dirs):
            f = os.path.join(d, stem + '_sidecar.json')
            if os.path.exists(f):
                sc = f; break
        if sc is None:
            bad.append(f'{os.path.basename(p)}: no generator sidecar'); continue
        try:
            gen = json.load(open(sc)).get('provenance', {}).get('generator', '')
        except Exception as ex:
            bad.append(f'{os.path.basename(p)}: sidecar unreadable ({type(ex).__name__})'); continue
        if not GEN.match(str(gen)):
            bad.append(f'{os.path.basename(p)}: sidecar provenance is not our generator ({gen!r})')
    return bad


def main(argv):
    if not argv:
        print(__doc__); return 2
    if argv[0] == '--list':
        images = [l.strip() for l in open(argv[1]) if l.strip()]; dirs = argv[2:]
    else:
        images = sorted(f for f in glob.glob(os.path.join(argv[0], '*')) if f.lower().endswith(IMG)
                        and not os.path.splitext(f)[0].endswith(SIDE)); dirs = argv[1:]
    bad = check(images, dirs)
    if bad:
        print(f'PUBLISH_GUARD_REFUSED {len(bad)} of {len(images)} inputs')
        for b in bad[:20]:
            print('  ' + b)
        return 3
    print(f'PUBLISH_GUARD_OK {len(images)} synthetic inputs')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
