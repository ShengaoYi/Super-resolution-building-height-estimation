import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data
from osgeo import gdal
import argparse
from SR.rrdbnet_arch import RealESRGAN
from mymodels import SRRegress_Cls_feature
from BH_loader import myImageFloder_S12_globe
from utils.preprocess import array2raster, array2raster_rio
import tifffile as tif
import rasterio

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=r'./data', help='Root directory for the data')
    parser.add_argument('--logdir', default=r'./weights/realesrgan_feature_aggre_weight_globe', help='Directory for model logs and weights')
    parser.add_argument('--logdirhr', default=r'./weights/realesrgan/checkpoint.tar', help='Pretrained RealESRGAN model path')
    parser.add_argument('--checkpoint', default='checkpoint20.tar', help='Checkpoint file for the model')
    parser.add_argument('--nchans', default=8, type=int, help='Number of channels for Sentinel-1 images')
    parser.add_argument('--nchanss2', default=6, type=int, help='Number of channels for Sentinel-2 images')
    parser.add_argument('--s1dir', type=str, default=r's1_test', help='Directory for Sentinel-1 images')
    parser.add_argument('--s2dir', type=str, default=r's2_test', help='Directory for Sentinel-2 images')
    parser.add_argument('--datastats', default='./datastatsglobe', help='Directory for data normalization statistics')
    parser.add_argument('--normmethod', default='minmax', help='Normalization method (e.g., meanstd, minmax)')
    parser.add_argument('--preweight', default='./datastatsglobe/bh_stats_usa.txt')
    parser.add_argument('--hir', type=int, nargs='+', default=[0, 3, 12, 21, 30, 60, 90, 255], help='Hierarchical range for height categories')
    args = parser.parse_args()
    return args

def predict_building_height(args):
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    logdir = args.logdir

    device = 'cuda'

    # Load pretrained models
    net_hr = RealESRGAN(pretrain_g_path=None,
                        pretrain_d_path=None,
                        device=device, scale=4,
                        num_block=23)
    net_hr.net_g.load_state_dict(torch.load(args.logdirhr)['net_g_ema'])
    net_hr.net_g.eval()
    for p in net_hr.net_g.parameters():
        p.requires_grad = False

    net = SRRegress_Cls_feature(encoder_name="efficientnet-b4",
                                in_channels=args.nchans, super_in=64,
                                super_mid=16, upscale=4,
                                chans_build=7).to(device)

    resume = os.path.join(logdir, args.checkpoint)

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
        if 'iter' in checkpoint.keys():
            start_epoch = checkpoint['iter']
        else:
            start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")
        return

    dataset = myImageFloder_S12_globe(rootname=args.datapath,
                                      datastats=args.datastats,
                                      normmethod='minmax',
                                      datarange=(0, 1),
                                      preweight=args.preweight,
                                      s1dir=args.s1dir,
                                      s2dir=args.s2dir,
                                      nchans=6,
                                      isaggre=True,
                                      ishir=True,
                                      hir=(0, 3, 12, 21, 30, 60, 90, 255)
                                      )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    srcgeotrans = dataset.geotrans
    nres = srcgeotrans[1] / 4.0  # Adjust resolution

    net.eval()
    net_hr.net_g.eval()

    output_dir = "./output_predictions3"
    output_building_dir = "./output_predictions_building3"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_building_dir, exist_ok=True)

    # 执行预测
    with torch.no_grad():
        for idx, (img, height, build, heightweight, img_path_list) in enumerate(tqdm(dataloader)):
            print(img_path_list)
            # 将输入数据加载到设备
            img = img.to(device, non_blocking=True)

            # 提取高分辨率的特征
            hr_fea = net_hr.net_g.forward_feature(img[:, :3])  # 假设输入的前三个通道是RGB

            # 通过网络进行建筑物高度预测
            ypred, build_pred = net.forward(img, hr_fea)

            # 将预测结果从GPU转移到CPU，并转换为numpy数组
            ypred = ypred.cpu().numpy()

            # 对预测结果进行后处理，剔除负值，并进行缩放
            ypred[ypred < 0] = 0  # 剔除负值
            ypred = np.round(ypred * 10).astype(np.uint16)  # 假设缩放因子为10

            build_pred = torch.softmax(build_pred, dim=1).cpu().numpy()
            build_pred = np.round(build_pred * 255).astype(np.uint16)

            # 累积预测结果
            n = img.shape[0]
            for i in range(n):

                # 从img_path提取文件名并设置输出路径
                img_path = img_path_list[i]
                basename = os.path.basename(img_path)
                output_height_tif = os.path.join(output_dir, f"{basename}")
                output_build_tif = os.path.join(output_building_dir, f"{basename}_build.tif")

                # 将当前图片的预测结果保存为tif
                ypred_single = ypred[i, 0]  # 单张图片的高度预测

                build_pred_single = build_pred[i]  # 单张图片的建筑物预测

                build_pred_single = np.argmax(build_pred_single, axis=0).astype(np.uint8)  # C H W -> H W

                array2raster_rio(output_build_tif, build_pred_single, src_tif=img_path, bands=1, nresolution=nres)

                # 使用 array2raster 函数保存结果
                array2raster(output_height_tif, ypred_single, src_tif=img_path, datatype=gdal.GDT_UInt16,
                             nresolution=nres, compressoption=['COMPRESS=DEFLATE', 'TILED=YES'])

            break


if __name__ == "__main__":
    args = get_args()
    predict_building_height(args)
