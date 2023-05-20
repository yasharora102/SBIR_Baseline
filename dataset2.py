import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
import itertools
import numpy as np
from glob import glob
from natsort import natsorted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FGSBIR_Dataset2(data.Dataset):
    def __init__(self, hp, mode):
        self.hp = hp
        self.mode = mode

        self.root_dir = os.path.join(hp.root_dir, hp.dataset_name)
        # for COCO dataset

        self.Sketch_dir = os.path.join(self.root_dir, "Sketch", self.mode.lower() if self.mode == "Train" else "val")
        self.GT_dir = os.path.join(self.root_dir, "GT", self.mode.lower() if self.mode == "Train" else "val")

        # import pdb

        # pdb.set_trace()

        # get the list of all the files in the directory of Tain SKetch

        self.Train_Sketch_subdirectories = os.listdir(self.Sketch_dir)

        # import pdb
        # remove .DS_Store from the list if preseent

        # if ".DS_Store" in self.Train_Sketch_subdirectories:
        #     self.Train_Sketch_subdirectories.remove(".DS_Store")

        # self.Train_GT_subdirectories = self.Train_Sketch_subdirectories.copy()

        # if ".DS_Store" in self.Train_GT_subdirectories:
        #     self.Train_GT_subdirectories.remove(".DS_Store")
        # make a dictionary of all the files in the directory of Train Sketch
        # key: subdirectory name
        # value: list of all the files in the subdirectory
        # from pdb import set_trace as stx

        # stx()
        self.Train_dict = {}
        self.GT_dict = {}
        
        self.Test_dict = {}
        self.Test_GT_dict = {}


        # self.Train_Sketch_images = []
        # self.Train_GT_images = []

        self.Sketch_images = natsorted(
            glob(os.path.join(self.Sketch_dir, "*", "*.png"))
        )

        self.GT_images = natsorted(
            glob(os.path.join(self.GT_dir, "*", "*.png"))
        )

        


        for subdirectory in self.Train_Sketch_subdirectories:
            # self.Train_dict[subdirectory] = os.listdir(
            #     os.path.join(self.Train_Sketch_dir, subdirectory)
            # )

            self.Train_dict[subdirectory] = natsorted(
                glob(os.path.join(self.Sketch_dir, subdirectory, "*.png"))
            )

            self.GT_dict[subdirectory] = natsorted(
                glob(os.path.join(self.GT_dir, subdirectory, "*.png"))
            )
         
         
        
          
        print("Total Sketch Images:", len(self.Sketch_images))
        print("Total GT Images:", len(self.GT_images))

        # import pdb; pdb.set_trace()

        self.train_transform = get_transform("Train")
        self.test_transform = get_transform("Test")
        # import pdb; pdb.set_trace()

    def __getitem__(self, item):
        # print("item: ", item)
        sample = {}
        if self.mode == "Train":
            # self.Train_GT_subdirectories = [x for x in os.listdir(self.Train_GT_dir)]
            # if ".DS_Store" in self.Train_GT_subdirectories:
            #     self.Train_GT_subdirectories.remove(".DS_Store")
            # import pdb; pdb.set_trace()
            sketch_path = self.Sketch_images[item]
            path = '/'.join(sketch_path.split('/')[-3:-1]) + '/' + sketch_path.split('/')[-1].split('.')[0]


            # only imafge name
            # positive_sample = self.Train_GT_images[item].split(".")[0]

            # import pdb; pdb.set_trace()

            positive_path = self.GT_images[item]
            positive_sample = os.path.splitext(os.path.split(positive_path)[-1])[0]
            positive_class = os.path.split(os.path.split(positive_path)[0])[-1]

            # get all the negative samples

            # 1.1 Positive class
            # 1.2 List of all classes
            # 1.3 Remove positive class from the list of all classes

            # import pdb; pdb.set_trace()
            possible_list = self.Train_Sketch_subdirectories.copy()
            possible_list.remove(positive_class)
            
            # negative_samples = [value for key, value in self.Train_dict.items() if key != positive_class]
            # negative_samples = list(itertools.chain.from_iterable(negative_samples))

            random_class = possible_list[randint(0, len(possible_list) - 1)]
            negative_samples = self.Train_dict[random_class]
            negative_path = negative_samples[randint(0, len(negative_samples) - 1)]

            # negative_path = negative_samples[randint(0, len(negative_samples) - 1)]
            negative_sample = negative_path.split("/")[-1].split(".")[0]


            sketch_img = Image.open(sketch_path).convert("RGB")
            positive_img = Image.open(positive_path).convert("RGB")
            negative_img = Image.open(negative_path).convert("RGB")

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {
                "sketch_img": sketch_img,
                "sketch_path": path,
                "positive_img": positive_img,
                "positive_path": positive_sample,
                "negative_img": negative_img,
                "negative_path": negative_sample,
            }
            import pdb; pdb.set_trace()

            # sketch_path = positive_path

        elif self.mode == "Test":
            # self.Test_GT_subdirectories = [x for x in os.listdir(self.Test_GT_dir)]
            # if ".DS_Store" in self.Test_GT_subdirectories:
            #     self.Test_GT_subdirectories.remove(".DS_Store")

            sketch_path = self.Sketch_images[item]
            path = '/'.join(sketch_path.split('/')[-3:-1]) + '/' + sketch_path.split('/')[-1].split('.')[0]

            # get class of the positive sample
            # cur_img = Image.open(sketch_path).convert("L")
            # # import pdb; pdb.set_trace()

            # vector_x = np.array(cur_img)

            # sketch_img = rasterize_Sketch(vector_x)
            sketch_img = Image.open(sketch_path).convert("RGB")
            sketch_img = self.test_transform(sketch_img)
            positive_sample = self.GT_images[item].split("/")[-1].split(".")[0]
            positive_path = self.GT_images[item]
            
            # positive_sample = self.Test_GT_images[item].split(".")[0]
            # positive_path = self.Test_GT_images_abs_pth[item]
            positive_img = self.test_transform(Image.open(positive_path).convert("RGB"))

            sample = {
                "sketch_img": sketch_img,
                "sketch_path": path,
                # "Coordinate": vector_x,
                "positive_img": positive_img,
                "positive_path": positive_sample,
            }

            # import pdb; pdb.set_trace()
        return sample

    def __len__(self):
        # if self.mode == "Train":
        return len(self.Sketch_images)
        # elif self.mode == "Test":
        #     return len(self.Test_Sketch_images)


def get_transform(type):
    transform_list = []
    if type == "Train":
        transform_list.extend([transforms.Resize((299, 299))])
    elif type == "Test":
        transform_list.extend([transforms.Resize((299, 299))])
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(transform_list)
