import os
import numpy as np
import tqdm
import os
import pandas as pd
from skimage import io
import numpy as np
import json
import tqdm
# R, G, B


def get_img_mean_std(scans_list, mode):
    means = []
    stds = []
    dirlist = []
    for scan in scans_list:
        for i in range(36):
            img_name = os.path.join('/home/qzp/3Ddata/RawData/scans', scan, 'color', '{}.jpg'.format(str(i)))
            print(img_name)
            img = io.imread(img_name)
            img = img
            assert img is not None, img_name + 'is not valid'
            # height*width*channels, axis=0 is the first dim
            mean = np.mean(np.mean(img, axis=0), axis=0)
            means.append(mean)
            std = np.std(np.std(img, axis=0), axis=0)
            stds.append(std)
    mean = np.mean(np.array(means), axis=0).tolist()
    std = np.mean(np.array(stds), axis=0).tolist()
    return {"mode":mode,'mean': mean, 'std': std}


if __name__ == '__main__':
    scans_list_train = ['snow_1', 'snow_12', 'snow_24', 'snow_25','snow_27','rock_2','rock_14','rock_15','rock_16','rock_17','plane_3','plane_13','plane_18','plane_19','plane_22']
    scans_list_eval = ['snow_27']
    getImgMeanStd = get_img_mean_std(scans_list_train,'train')
    print(getImgMeanStd)