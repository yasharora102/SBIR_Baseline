import argparse
from dataset2 import FGSBIR_Dataset2

# import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


#  input_ = TF.to_tensor(img).unsqueeze(0).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import FGSBIR_Model
import time
import numpy as np
import wandb

parser = argparse.ArgumentParser(description="Fine-Grained SBIR Model")

parser.add_argument("--dataset_name", type=str, default="COCO")
parser.add_argument(
    "--backbone_name", type=str, default="VGG", help="VGG / InceptionV3/ Resnet50"
)
parser.add_argument(
    "--pool_method",
    type=str,
    default="AdaptiveAvgPool2d",
    help="AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d",
)
parser.add_argument("--root_dir", type=str, default=os.getcwd())
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--nThreads", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--max_epoch", type=int, default=200)
parser.add_argument("--eval_freq_iter", type=int, default=100)
parser.add_argument("--print_freq_iter", type=int, default=1)
parser.add_argument("--logs", type=bool, default=False)

hp = parser.parse_args()

logs = hp.logs

if logs:
    wandb.init(
        # set the wandb project where this run will be logged
        project="sbir",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": hp.learning_rate,
        "architecture": "FGSBIR",
        "dataset": "SketchyCoco",
        "epochs": hp.max_epoch,
        }
    )

dataset_Train = FGSBIR_Dataset2(hp, mode="Train")
dataloader_Train = data.DataLoader(
    dataset_Train, batch_size=hp.batchsize, shuffle=True, num_workers=0
)

dataset_Test = FGSBIR_Dataset2(hp, mode="Test")
# dataloader_Test = data.DataLoader(
#     dataset_Test, batch_size=hp.batchsize, shuffle=False, num_workers=hp.nThreads
# )
dataloader_Test = data.DataLoader(
    dataset_Test, batch_size=hp.batchsize, shuffle=False, num_workers=0
)
print(hp)

model = FGSBIR_Model(hp)
model.to(device)
step_count, top1, top10 = -1, 0, 0

for i_epoch in range(hp.max_epoch):
    train_loss = 0
    for batch_data in dataloader_Train:
        step_count = step_count + 1
        # print("step_count: ", step_count)
        start = time.time()
        # model.train()
        loss = model.train_model(batch=batch_data)

        train_loss = train_loss + loss

        if step_count % hp.print_freq_iter == 0:
            print(
                "Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}".format(
                    i_epoch, step_count, loss, top1, top10, time.time() - start
                )
            )

        if (step_count % hp.eval_freq_iter == 0) and (step_count > 0):
            print("Evaluation")
            with torch.no_grad():
                top1_eval, top10_eval = model.evaluate(dataloader_Test)
                print("results : ", top1_eval, " / ", top10_eval)
                if logs:
                    wandb.log({"top1_eval": top1_eval, "top10_eval": top10_eval})


            if top1_eval > top1:
                # torch.save(model.state_dict(), hp.backbone_name + '_' + hp.dataset_name + '_model_best.pth')
                top1, top10 = top1_eval, top10_eval
                print("Model Updated")

    if logs:
        wandb.log({"train_loss": train_loss})

if logs:
    wandb.finish()
