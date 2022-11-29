import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from unet import UNET
from unet_dilated import UNET as DUNET
from model.build_bisenet import BiSeNet
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from utils import save_images
import json
from cityscapes import Cityscapes
from eval import batch_intersection_union, pixelAccuracy
import argparse
import sys

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser('train u-net network for semantic segmentation')
parser.add_argument('--epochs', help='the number of epochs to train the model', type=int, default=30)
parser.add_argument('--start_epoch', help='the number of the first epoch (useful for restarting from a checkpoint)', type=int, default=0)
parser.add_argument('--softmax_layer', help='whether to add the softmax layer at the end; default is true', type=boolean_string, default=True)
parser.add_argument('--dilation',help='whether to use DilatedNet (True) or UNet (False)', type=boolean_string, default=False)
parser.add_argument('--checkpoint_path',help='path for saving the best model', type=str, default="output/segmentation/checkpoint/")
parser.add_argument('--output_path',help='path for saving the predictions of the model', type=str, default="output/segmentation/images/")
parser.add_argument('--tensorboard_logdir',help='path for saving the runs data for tensorboard', type=str, default="output/segmentation/runs/")
parser.add_argument('--save_images_step',help='step for saving predictions output during validation', type=int, default=1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

crop_width = 512
crop_height = 256
composed = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(p=0.5), T.RandomAffine(0, scale=[0.75, 2.0]), T.RandomCrop((crop_height, crop_width), pad_if_needed=True)])
train_data = Cityscapes("./data/Cityscapes", "images/", "labels/", train=True, info_file="info.json", transforms=composed)
val_data = Cityscapes("./data/Cityscapes", "images/", "labels/", train=False, info_file="info.json", transforms=composed)

def val(args, model, dataloader, validation_run):
    print(f"{'#'*10} VALIDATION {'#' * 10}")
    val_size = len(dataloader)
    #prepare info_file to save examples
    info = json.load(open("data/Cityscapes/info.json"))
    palette = {i if i!=19 else 255:info["palette"][i] for i in range(20)}
    #mean = torch.as_tensor(info["mean"]).to(device)
    
    with torch.no_grad():
        model.eval() #set the model in the evaluation mode
        inter_record = 0
        union_record = 0
        pixel_acc_record = 0

        for i, (image, label) in enumerate(tqdm(dataloader)): 
            label = label.type(torch.LongTensor)
            label = label.long().to(device)
            image = image.to(device)

            #get RGB predict image
            predict = model(image)

            # compute intersection and union metrics
            inter, union = batch_intersection_union(predict, label, 19)
            inter_record += inter
            union_record += union
            #compute per pixel accuracy
            pixel_acc_record += pixelAccuracy(torch.argmax(predict, dim=1), label)

            #Save the image
            if args.output_path is not None and i % args.save_images_step == 0: 
                os.makedirs(args.output_path, exist_ok=True)
                output_prediction = torch.argmax(predict, dim=1).cpu().numpy() #if args.softmax_layer else torch.clamp(predict, min=0, max=18).to(torch.uint8).cpu().numpy()
                save_images(palette, predict=output_prediction, path_to_save=f"{args.output_path}img_{validation_run}_{i}.png")
    
    precision = pixel_acc_record/val_size
    per_class_mIoU = inter_record/union_record
    for (c, IoU) in enumerate(per_class_mIoU):
        print(f"Class = {c}: IoU = {IoU}")
    total_inter = inter_record.sum()
    total_union = union_record.sum()
    total_mIoU = total_inter/total_union
    print('precision per pixel for test: %.3f' % precision)
    print('mIoU for validation: %.3f' % total_mIoU)
    return precision, total_mIoU #precision, miou 


def train(args, model, optimizer, train_loader, valloader, batch_size=4, validation_step=1):             
    
    #Set the loss of G
    if args.softmax_layer == True:
        model_name = "unet_ce" if args.dilation == False else "dilated_ce"
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    else:
        model_name = "unet_mse" if args.dilation == False else "dilated_mse"
        loss_func = torch.nn.MSELoss()
    #Writer
    writer = SummaryWriter(f"{args.tensorboard_logdir}{model_name}")
    
    max_miou = 0
    step = 0

    for epoch in range(args.start_epoch, args.epochs):

        #Set the model to train mode
        model.train()

        #TQDM Setting
        tq = tqdm(total = len(train_loader) * batch_size) 
        tq.set_description('epoch %d' % (epoch))
        
        #Loss array
        loss_seg_record = [] 

        for (images, labels) in train_loader:
            #Train with source
            labels = labels.long() if args.softmax_layer else labels.float()
            images = images.to(device)
            labels = labels.to(device)
            # print(f"lables: {images}")
            optimizer.zero_grad()

            output = model(images)
                
            # print(f"logits: {output}")
            loss_seg = loss_func(output.squeeze(), labels)                                             

            loss_seg.backward() 

            optimizer.step()
          
            #Print statistics
            tq.update(batch_size)
            tq.set_postfix({"loss_seg" : f'{loss_seg:.6f}'})
            step += 1
            writer.add_scalar('loss_seg_step', loss_seg, step)
            loss_seg_record.append(loss_seg.item())

    
        tq.close()

        #Loss_seg
        loss_train_seg_mean = np.mean(loss_seg_record)
        writer.add_scalar('epoch/loss_epoch_train_seg', float(loss_train_seg_mean), epoch)
        print(f'Average loss_seg for epoch {epoch}: {loss_train_seg_mean}')
        
        #Validation step
        if epoch % validation_step == 0:
                precision, miou = val(args, model, valloader, epoch)
                #Check if the current model is the best one
                if miou > max_miou:
                    max_miou = miou
                    os.makedirs(args.checkpoint_path, exist_ok=True)
                    torch.save(model.state_dict(),
                            os.path.join(args.checkpoint_path, f"best_model_{model_name}.pth"))

                writer.add_scalar('epoch/precision_val', precision, epoch)
                writer.add_scalar('epoch/overall miou val', miou, epoch)
                print(f"Validation precision: {precision}")
                print(f"Validation miou: {miou}")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

def main():
    batch_size = 4
    lr = 1e-3
    weight_decay = 0.0005
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    # if args.softmax_layer == True:
    #     if args.dilation == True:
    #         model = DUNET(3, 19).to(device)
    #     else:
    #         model = UNET(3,19).to(device)
    # else:
    #     # We directly let the U-Net to output the correct label, instead of logits
    #     if args.dilation == True:
    #         model = DUNET(3, 1).to(device)
    #     else:
    #         model = UNET(3, 1).to(device)
    
    model = BiSeNet(19, "resnet101").to(device)
    model.module.load_state_dict(torch.load("/content/drive/MyDrive/MLDL_Project/BiSeNet/models/29May_soloresize_test2/latest_dice_loss.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val(args, model, val_data, 0)
    #train(args, model, optimizer, train_loader, val_loader)

if __name__ == '__main__':
    main()