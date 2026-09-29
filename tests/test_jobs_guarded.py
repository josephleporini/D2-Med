"""G3: every pod job that pushes to a repository must run the publish guard (infra/guard_public.py). Jobs written before
the guard existed are listed; they read only the synthetic splits hard-wired in them and must not be pointed at other
data. A new job is not allowed on this list."""
import os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = {'bt1.sh', 'gpu_timing.sh', 'l2_dev5.sh', 'publish_results.sh', 'relief_oracle.sh', 'render_splits.sh',
          'tta_flip.sh', 'render_gpu.sh', 'pilot_gpu.sh'}


def test_pushing_jobs_call_the_guard():
    missing = []
    for f in sorted(glob.glob(os.path.join(HERE, '..', 'jobs', '*.sh'))):
        s = open(f).read(); name = os.path.basename(f)
        if 'git push' in s and 'guard_public.py' not in s and name not in LEGACY:
            missing.append(name)
    assert not missing, f'jobs push without the publish guard: {missing}'
