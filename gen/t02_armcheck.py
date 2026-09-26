import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import bpy
from mathutils import Vector
import manikin as mk
for pose in ['rest','neutral','abducted','arm_overhead','arm_across']:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ob, arm, j, s = mk.build()
    if pose!='rest': mk.limb_pose(arm, pose)
    out=[]
    for side in 'LR':
        sh = arm.pose.bones['upperarm01.'+side].head; wr = arm.pose.bones['wrist.'+side].head
        d = (wr - sh).normalized()
        # angle below horizontal (0 = horizontal, 90 = straight down), lateral = outward component
        lat = d.x if side=='L' else -d.x
        out.append(f"{side}: down {math.degrees(math.asin(-d.z)):5.0f} deg, outward {lat:+.2f}, forward {-d.y:+.2f}")
    print(f"{pose:13s}", ' | '.join(out))
