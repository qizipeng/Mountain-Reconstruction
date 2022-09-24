from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from PIL import Image
import os
img = io.imread('./SLIC/30.jpg')
img = resize(img,(128,128))
#print(img.shape)
segments = slic(img, n_segments=20, compactness=20,enforce_connectivity=True,convert2lab=True)
#segments = np.expand_dims(segments, -1)
#print(segments.shape)
#segments = np.int8(segments)
#seg = Image.fromarray(segments)
#seg = seg.convert('L') 
#seg.show()
#name = im.split(".")[0]
#seg.save(os.path.join(outputdir,name+".png"))



#n_liantong=segments.max()+1
#print('n_liantong:',n_liantong)
#area=np.bincount(segments.flat)
#w,h=segments.shape
#print(area/(w*h))
#print((max(area/(w*h))),(min(area/(w*h))))
 
out=mark_boundaries(img,segments)
plt.subplot(111)
plt.imshow(out)
plt.show()

