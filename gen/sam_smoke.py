import time, numpy as np, torch, cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(2)
t=time.time()
m=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml','../models/sam2_1_hiera_tiny.pt',device='cpu')
P=SAM2ImagePredictor(m); print('load s',round(time.time()-t,1))
img=cv2.cvtColor(cv2.imread('../out/check20/C001.jpg'),cv2.COLOR_BGR2RGB)
t=time.time(); P.set_image(img); print('encode s',round(time.time()-t,1))
z=np.load('../out/check20/C001_rtmw133.npz'); k=z['k']
pts=np.array([k[7],k[9]]); lab=np.array([1,1])
t=time.time(); masks,scores,_=P.predict(point_coords=pts,point_labels=lab,multimask_output=True); print('predict s',round(time.time()-t,2),'scores',scores.round(2),'areas',[int(x.sum()) for x in masks])
