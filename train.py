
import sys
import datetime
import torch.optim
import os
import argparse
import time
import dataloader
from SSIM import SSIM,SSIMLOSS
import torchvision
import matplotlib.pyplot as plt
# import VGG
import numpy as np
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import threading
import subprocess
import datetime
from util import *
from losses import *

from fdce_net import *


def inplace_relu(m):  # 缓解显存压力
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True



def train(config):
    UIE_net = fdce_net().cuda()

    train_dataset = dataloader.UIE_loader(config.orig_images_path, config.hazy_images_path, shrink=2,mode="train")
    val_dataset = dataloader.UIE_loader(config.orig_images_path_val, config.hazy_images_path_val, shrink=2, mode="val")
    # todo: change dataloder settings
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True, drop_last=True)

    L1 = nn.SmoothL1Loss()
    criterion = SSIMLOSS()
    comput_ssim = SSIM()
    Perceptual = PerceptualLoss()

    UIE_net.train()

    zt = 1
    Iters = 0
    iter_loss = []  # 计数损失曲线用
    indexX = []  # 计数损失曲线用
    indexY = []
    # config.lr = 0.0003
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()
    criterion_tv = TVLoss()
    for epoch in range(1, config.num_epochs+1):

        if epoch <= 3:
            config.lr = 0.0003
        elif epoch > 3 and epoch <= 15:
            config.lr = 0.0001
        elif epoch > 15 and epoch <= 30:
            config.lr = 0.00006
        elif epoch > 30 and epoch <= 40:
            config.lr = 0.00003
        elif epoch > 40 and epoch <= 50:
            config.lr = 0.00001
        elif epoch > 50 and epoch <= 60:
            config.lr = 0.000006
        elif epoch > 60 and epoch <= 70:
            config.lr = 0.000003
        elif epoch > 70 and epoch <= 200:
            config.lr = 0.000001

        optimizer = torch.optim.AdamW(UIE_net.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=0.02)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        for iteration, (img_clean, img_haze, extension) in loop:

            img_clean = img_clean.cuda()
            img_haze = img_haze.cuda()

            gray, hist, result = UIE_net(img_haze)

            loss = 0
            loss += criterion(result, img_clean)
            loss += L1(result, img_clean)
            loss += 0.5*nn.functional.l1_loss(hist, get_color_hist(result).cuda())
            loss += 0.05*Perceptual(result, img_clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ssim = criterion(result, img_clean)
            # todo: change these code

            iter_loss.append(loss.item())
            if Iters % config.display_iter == 0:
                writer.add_scalar("train", 1-loss_ssim, global_step=Iters)

            Iters += 1

            loop.set_description(f'Epoch [{epoch}/{config.num_epochs}]LR=%f'%config.lr)
            loop.set_postfix(
                Totaloss=loss.item(),
            )

        _ssim = []
        #print("start Val!")
        # Validation Stage
        time.sleep(0.5)
        mkdir(config.sample_output_folder + f"/epochs/epoch{str(epoch).zfill(3)}")
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader), ncols=60)
            for iteration1, (img_clean, img_haze, extension) in loop_val:

                img_clean = img_clean.cuda()
                img_haze = img_haze.cuda()
                gray, color, result = UIE_net(img_haze)
                # result = UIE_net(img_haze)
                _s = comput_ssim(result, img_clean)
                _ssim.append(_s.item())
                displayImg = torch.cat((img_haze, img_clean,catDissplayImg(result)), 0)
                # todo: change model return
                torchvision.utils.save_image(displayImg,f"{config.sample_output_folder}/epochs/epoch{str(epoch).zfill(3)}/{str(iteration1 + 1).zfill(3)}{extension[0]}")

            _ssim = np.array(_ssim)

            ssim_log = "[%i,%f]" % (epoch, np.mean(_ssim)) + "\n"


            indexX.append(epoch)
            now = np.mean(_ssim)
            writer.add_scalar('val', now, global_step=epoch)
            if indexY == []:
                indexY.append(now)
                print("\033[31m First epoch:\033[0m", now)
                torch.save(UIE_net.state_dict(), config.snapshots_folder + 'best.pth')
            else:
                now_max = np.argmax(indexY)
                indexY.append(now)
                print('\033[31m max epoch %i\033[0m' % (now_max+1),
                      '\033[31m SSIM:\033[0m', indexY[now_max],
                      '\033[31m Now Epoch mean SSIM is:\033[0m', now)
                if now >= indexY[now_max]:
                    ssim_log = ssim_log[:-1] + "*" + ssim_log[-1:]
                    print("\033[31m Now epoch is best ！！\033[0m")
                    loop_dis = tqdm(enumerate(val_loader), total=len(val_loader), ncols=60)
                    for iteration1, (img_clean, img_haze, extension) in loop_dis:
                        # print("va1 : %s" % str(iteration1))

                        img_clean = img_clean.cuda()
                        img_haze = img_haze.cuda()
                        gray, color, result = UIE_net(img_haze)
                        # result = UIE_net(img_haze)
                        # img_clean, img_haze, result = F.interpolate(img_clean, og_size, mode='bilinear'),F.interpolate(img_haze, og_size, mode='bilinear'),F.interpolate(result, og_size, mode='bilinear')

                        displayImg = torch.cat(
                            (img_haze, img_clean, catDissplayImg(result)), 0)
                        # todo: change these code
                        grid = torchvision.utils.make_grid(displayImg,normalize=True,range=(0,1))
                        writer.add_image('best result', grid, iteration1)

                        torchvision.utils.save_image(displayImg,f"{config.sample_output_folder}/best/{str(iteration1 + 1).zfill(3)}{extension[0]}")
                        torchvision.utils.save_image(result,f"{config.results_output_folder}{str(iteration1 + 1).zfill(3)}{extension[0]}")
                    torch.save(UIE_net.state_dict(), config.snapshots_folder + 'best.pth')
                else:
                    shutil.rmtree(config.sample_output_folder + f"/epochs/epoch{str(epoch).zfill(3)}")
            # with open("trainlog/%s.log" % (config.modelname), "a+",
            with open('training_data/%s/train.log' % config.modelname, "a+",
                      encoding="utf-8") as f:
                f.write(ssim_log)
    torch.save(UIE_net.state_dict(), config.snapshots_folder + "%s.pth"%config.modelname)

if __name__ == "__main__":

    defaultname = "UIEB"+datetime.datetime.now().strftime("%m%d-%H%M")
    # todo: change defaultname
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=20)
    parser.add_argument('--snapshots_folder', type=str, default="training_data/%s/weight/" % defaultname)
    parser.add_argument('--sample_output_folder', type=str, default="training_data/%s/sample/" % defaultname)
    parser.add_argument('--results_output_folder', type=str, default="training_data/%s/sample/results/" % defaultname)
    parser.add_argument('--modelname', type=str, default=defaultname)
    parser.add_argument('--recurrent_iter', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.9)


    parser.add_argument('--orig_images_path', type=str, default=r"/home/share/UIE_Datasets/UIEB-100/ref-790")
    parser.add_argument('--hazy_images_path', type=str, default=r"/home/share/UIE_Datasets/UIEB-100/raw-790")
    parser.add_argument('--orig_images_path_val', type=str,default=r"/home/share/UIE_Datasets/UIEB-100/ref-100")
    parser.add_argument('--hazy_images_path_val', type=str,default=r"/home/share/UIE_Datasets/UIEB-100/raw-100")

    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--cudaid', type=str, default="0", help="choose cuda device id 0-7.")
    # todo: change cudaid and batchsize
    config = parser.parse_args()


    mkdir("training_data")

    # 把未正常训练的文件夹删除掉
    previous_training = os.listdir("training_data")
    for src_folder in previous_training:
        src_folder = os.path.join("training_data", src_folder)
        train_log_file = os.path.join(src_folder, "train.log")
        if os.path.exists(train_log_file):
            with open(train_log_file, 'r') as file:
                line_count =  sum(1 for line in file)
            if line_count < 7:
                try:
                    shutil.rmtree(src_folder)
                    print(f"文件夹 '{src_folder}' 已删除。")
                except OSError as e:
                    print(f"删除文件夹 '{src_folder}' 失败: {e}")
        else:
            # print(f"文件 '{train_log_file}' 不存在。")
            pass



    mkdir("training_data/%s" % config.modelname)
    mkdir("training_data/%s/project_files" % (config.modelname))
    mkdir(config.snapshots_folder)
    mkdir(config.sample_output_folder)
    mkdir(config.sample_output_folder + "/best")
    mkdir(config.sample_output_folder + "/epochs")
    mkdir(config.results_output_folder)
    with open("training_data/%s/train.log" % config.modelname, 'w') as file:
        pass
    with open("training_data/%s/记录.txt" % config.modelname, 'w') as file:
        pass



    current_time = datetime.datetime.now()

    with open('training_data/%s/project_files/%s.txt' % (config.modelname, config.modelname), "w") as f:  # 设置文件对象
        for i in vars(config):
            f.write(i + ":" + str(vars(config)[i]) + '\n')
        f.write("train time:%s"% str(current_time))

    # transfPY(config)
    copy_project_files(os.getcwd(), 'training_data/%s/project_files' % config.modelname)
    # thread = threading.Thread(target=tbDisplay)
    # thread.start()
    time.sleep(2)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cudaid

    try:
        train(config)
        sys.exit()

    except Exception as e:
        raise(e)
        torch.cuda.empty_cache()
        time.sleep(2)
        sys.exit()

