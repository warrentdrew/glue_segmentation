from preprocess import define_region
import cv2
import numpy as np
from model import unet_bn, unet_bn_deep
import matplotlib.pyplot as plt
import os
from preprocess import get_center_key_from_imgname
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ''
def hardmax(arr):
    arr[arr >= 0.5] = 1
    arr[arr < 0.5] = 0
    return arr


def prepare_pred_data(datapath, cropsize, center, offset, model, margin = 32):
    img = cv2.imread(datapath, 0)

    h, w  = img.shape
    start = center - offset
    end = center + offset
    res = np.zeros([h, w, 1])
    x_pred = []
    for x in range(margin, h - margin - cropsize + 1, cropsize):
        for y in range(start, end, cropsize):
            #print(x, y)

            imgshow = img[x: x + cropsize, y: y + cropsize]
            img2 = (imgshow - 127.0) / 255.0
            img_to_pred = img2.reshape((cropsize, cropsize, 1))
            #cv2.imshow('data', imgshow)
            x_pred.append(img_to_pred)

    x_pred_arr = np.array(x_pred)
    #print(x_pred_arr.shape)
    pred = hardmax(model.predict(x_pred_arr, batch_size = x_pred_arr.shape[0]))
    i = 0
    for x in range(margin, h - margin - cropsize + 1, cropsize):
        for y in range(start, end, cropsize):
            res[x: x + cropsize, y: y + cropsize, :] = pred[i, ...]
            i += 1
    #print('sum result', np.sum(res))
    return res

def prepare_pred_data_mt(datapath, cropsize, center, offset, model, margin = 32):
    img = cv2.imread(datapath, 0)

    h, w  = img.shape
    start = center - offset
    end = center + offset
    res = np.zeros([h, w, 1])
    x_pred = []
    for x in range(margin, h - margin - cropsize + 1, cropsize):
        for y in range(start, end, cropsize):
            #print(x, y)

            imgshow = img[x: x + cropsize, y: y + cropsize]
            img2 = (imgshow - 127.0) / 255.0
            img_to_pred = img2.reshape((cropsize, cropsize, 1))
            #cv2.imshow('data', imgshow)
            x_pred.append(img_to_pred)

    x_pred_arr = np.array(x_pred)
    #print(x_pred_arr.shape)
    pred = hardmax(model.predict(x_pred_arr, batch_size = x_pred_arr.shape[0]))
    i = 0
    for x in range(margin, h - margin - cropsize + 1, cropsize):
        for y in range(start, end, cropsize):
            res[x: x + cropsize, y: y + cropsize, :] = pred[i, ...]
            i += 1
    #print('sum result', np.sum(res))
    return res

def colorize(prediction, colors={0 : np.array([0,0,0]),         #class 0: background
                                 1 : np.array([0.2,1,0])}):     #class 1: glue (in green mask)

    #prediction here need to have dim 3
    pred_picture = np.zeros(shape= prediction.shape[:2] + (3,))
    for x , row in enumerate(prediction):
        for y, col in enumerate(row):
                pred_picture[x, y, :] = colors[int(prediction[x,y,...])]

    return pred_picture

def mosaic(datapath, mask):
    img = cv2.imread(datapath, 0) #shape h, w
    colormask = colorize(mask) #shape h,w,3

    fig = plt.figure(dpi=100)
    fig.suptitle('Prediction')

    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(img, interpolation='none', cmap='gray')
    ax.imshow(colormask, interpolation='none', cmap='gray', alpha=0.3)

    return plt

def pred_vs_gt(datapath, pred, gtpath, bmaskpath):
    img = cv2.imread(datapath, 0)  # shape h, w
    colorpred = colorize(pred)  # shape h,w,3
    gt = cv2.imread(gtpath, 0)
    gt[gt == 255] = 1
    colorgt = colorize(gt)
    bmask = cv2.imread(bmaskpath, 0)
    bmask[bmask == 255] = 1
    colorbmask = colorize(bmask, colors={0 : np.array([0,0,0]),      #class 0: background
                                         1 : np.array([1, 1, 0.2])})   # yellow

    fig = plt.figure(dpi=100, num = 'pred vs gt', figsize= (18,7))
    #fig.suptitle('pred vs gt')
    #plt.title('abc')

    ax1 = fig.add_subplot(121)
    ax1.set_title('pred')
    ax1.axis('off')
    ax1.imshow(img, interpolation='none', cmap='gray')
    ax1.imshow(colorpred, interpolation='none', cmap='gray', alpha=0.3)
    ax1.imshow(colorbmask, interpolation='none', cmap='gray', alpha=0.2)

    ax2 = fig.add_subplot(122)
    ax2.set_title('gt')
    ax2.axis('off')
    ax2.imshow(img, interpolation='none', cmap='gray')
    ax2.imshow(colorgt, interpolation='none', cmap='gray', alpha=0.3)
    ax2.imshow(colorbmask, interpolation='none', cmap='gray', alpha=0.2)
    fig.set_tight_layout(True)
    #plt.savefig('resampled_focal.jpg')
    return plt

def savepath_from_img_path(imgpath, suffix):
    basename = os.path.basename(imgpath)
    name = os.path.splitext(basename)[0]
    savename = name + "_{}.jpg".format(suffix)
    rtpath = os.path.dirname(imgpath)
    savepath = os.path.join(rtpath, savename)
    #print(savepath)
    return savepath

def pred_whole(model, cropsize = 160, margin = 0, rt_dir = '/home/zhuyipin/DATASET/segmentation/dataset/test/', rt_bmask = '/home/zhuyipin/DATASET/segmentation/Mask/'):
    for subdir in os.listdir(rt_dir):
        subdirpath = os.path.join(rt_dir, subdir)
        for subsubdir in os.listdir(subdirpath):
            subsubdirpath = os.path.join(subdirpath, subsubdir)
            for file in os.listdir(subsubdirpath):
                if file.find('_mask') != -1:
                    maskpath = os.path.join(subsubdirpath, file)
                    imgpath = maskpath.replace('_mask', '')
                    bmaskname = get_center_key_from_imgname(imgpath)
                    machine = imgpath.split('/')[7]
                    bmaskdir = os.path.join(rt_bmask, '{}_MASK'.format(machine))
                    dict = define_region(maskdir=bmaskdir)
                    t0 = time.time()
                    if machine == 'AS22':
                        offset = 240
                    else:
                        offset = 320
                    #print("offset", offset, machine)
                    res = prepare_pred_data(imgpath, cropsize = cropsize, center = dict[bmaskname],
                                      offset = offset, model = model, margin = margin)
                    #res = pred_no_crop(model, imgpath, center = dict[bmaskname], offset = 320)
                    #print('max', np.max(res))
                    #cv2.imshow("res", res * 255)
                    #cv2.waitKey(0)
                    print("prediction time: {:.3f}".format(time.time() - t0))
                    bmaskpath = os.path.join(bmaskdir, bmaskname)
                    #print('aa',imgpath)
                    #print('bb', bmaskpath)
                    pred_vs_gt(imgpath, res, maskpath, bmaskpath)
                    savepath = savepath_from_img_path(imgpath, suffix = "shallow32")
                    #print(savepath)
                    plt.savefig(savepath)




def resize_pred_data(model, datapath, height, width):
    img = cv2.imread(datapath, 0)
    img = cv2.resize(img, (width, height))
    img = (img - 127.0) / 255.0
    img = img.reshape((1,) + img.shape + (1,))
    res = hardmax(model.predict(img, batch_size = 1))
    return res

def pred_no_crop(model, datapath, center, offset = 320):
    img = cv2.imread(datapath, 0)
    img = (img - 127.0) / 255.0
    imgcrop = img[:, center - offset : center + offset]
    #print("imgcrop shape:", imgcrop.shape)
    imgcrop = imgcrop.reshape((1,) + imgcrop.shape + (1,))

    t0 = time.time()
    res = hardmax(model.predict(imgcrop, batch_size = 1))
    print("time elapsed:", time.time() - t0)
    ret = np.zeros([960, 1280])
    ret[:, center - offset : center + offset] = np.squeeze(res)

    return ret





if __name__ == '__main__':
    # datapath = "/home/zhuyipin/DATASET/segmentation/dataset/AS23/01395344/01395344_ASBLD0901_AS23_3_narrow.bmp"
    # maskpath = "/home/zhuyipin/DATASET/segmentation/dataset/AS23/01395344/01395344_ASBLD0901_AS23_2_narrow_mask.bmp"
    #
    # datapath2 = '/home/zhuyipin/DATASET/segmentation/dataset/test/AS23/01409546/01409546_ASBLD0901_AS23_5_OK.bmp'
    # maskpath2 = "/home/zhuyipin/DATASET/segmentation/dataset/test/AS23/01409546/01409546_ASBLD0901_AS23_5_OK_mask.bmp"
    #
    # cropsize = 160
    # margin = 0
    # bmaskname = 'f5_C_mask.bmp'
    # bmaskdir = '/home/zhuyipin/DATASET/segmentation/Mask/AS23_MASK'
    # model = unet_bn(input_size=(cropsize, cropsize, 1))
    # #model.load_weights('unet_resnet_correct_val_no_gen_jcd.hdf5')
    # model.load_weights('unet_resnet_resize.hdf5')
    # dict = define_region(maskdir=bmaskdir)
    # #print(dict)
    #
    #
    # res = prepare_pred_data(datapath2, cropsize = cropsize, center = dict[bmaskname], offset = 320, model = model, margin = margin) # 467
    #
    #
    # bmaskpath = os.path.join(bmaskdir, bmaskname)
    # #mosaic(datapath2, res)
    # pred_vs_gt(datapath2, res, maskpath2, bmaskpath)
    # plt.show()




    
    height = 960
    width = 640
    cropsize = 160
    model = unet_bn(input_size=(cropsize, cropsize, 1), init_channel = 32)
    #model.load_weights('unet_resnet_correct_val_no_gen_jcd.hdf5')
    model.load_weights('unet_resnet_shallow32.hdf5')
    pred_whole(model)





    """

    datapath = "/home/zhuyipin/DATASET/segmentation/dataset/test/AS23/01394179/01394179_ASBLD0901_AS23_1_ok.bmp"
    maskpath = "/home/zhuyipin/DATASET/segmentation/dataset/test/AS23/01394179/01394179_ASBLD0901_AS23_1_ok_mask.bmp"
    bmaskname = 'f1_C_mask.bmp'
    bmaskdir = '/home/zhuyipin/DATASET/segmentation/Mask/AS23_MASK'

    model = unet_bn(input_size=(960, 640, 1))
    model.load_weights('unet_resnet_resize.hdf5')

    dict = define_region(maskdir=bmaskdir)
    res = pred_no_crop(model, datapath, center = dict[bmaskname], offset = 320)

    print(res.shape)
    #res = res.reshape((1280, ))
    #print("max:", np.max(res))
    bmaskpath = os.path.join(bmaskdir, bmaskname)
    pred_vs_gt(datapath, res, maskpath, bmaskpath)
    plt.show()

    """