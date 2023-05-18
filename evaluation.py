import time
import torch.nn as nn
from torch import optim
import  torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
import argparse
from dataset import get_dataloader
from model import FGSBIR_Model
import torchvision.transforms as transforms
from rasterize import mydrawPNG_fromlist, get_stroke_num
from itertools import combinations

def Evaluate_FGSBIR(model, datloader_Test):

    sketch_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    Image_Feature_ALL = []
    Image_Name = []
    Sketch_Feature_ALL = []
    Sketch_Name = []
    start_time = time.time()

    for i_batch, sanpled_batch in enumerate(datloader_Test):
        sketch_feature, positive_feature= model.test_forward(sanpled_batch)
        Sketch_Feature_ALL.extend(sketch_feature)
        Sketch_Name.extend(sanpled_batch['sketch_path'])
        print(i_batch)

        for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            if positive_name not in Image_Name:
                Image_Name.append(sanpled_batch['positive_path'][i_num])
                Image_Feature_ALL.append(positive_feature[i_num])

    rank = torch.zeros(len(Sketch_Name))
    Image_Feature_ALL = torch.stack(Image_Feature_ALL)
    Sketch_Feature_ALL = []

    for i_batch, sanpled_batch in enumerate(datloader_Test):


        sketch_coord = sanpled_batch['Coordinate'][0].numpy()
        total_stroke = get_stroke_num(sketch_coord)

        stroke_idx_list = list(range(total_stroke))
        stroke_combi_all = []
        for x in range(1, total_stroke+1):
            stroke_combi_all.extend(list(combinations(stroke_idx_list, x)))

        rank_sketch = []
        print(i_batch, len(stroke_combi_all))

        for idx in range(len(stroke_combi_all) // 128 + 1):

            if (idx + 1) * 128 <= len(stroke_combi_all):
                stroke_combi = stroke_combi_all[idx * 128: (idx + 1) * 128]
            else:
                stroke_combi = stroke_combi_all[idx * 128: len(stroke_combi_all)]
            print(idx)


            sketch_image = [sketch_transform(mydrawPNG_fromlist(sketch_coord, x)) for x in stroke_combi]
            sketch_image = torch.stack(sketch_image, dim=0)

            s_name = sanpled_batch['sketch_path'][0]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            sketch_feature_combi = model.sample_embedding_network(sketch_image.to(device)).cpu()



            for sketch_feature in sketch_feature_combi:
                target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
                rank_sketch.append(distance.le(target_distance).sum())

        rank[i_batch] = torch.stack(rank_sketch).min()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('Time to EValuate:{}'.format(time.time() - start_time))
    return top1, top10



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='./../Dataset/')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=10)

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    model = FGSBIR_Model(hp)
    model.to(device)
    model.load_state_dict(torch.load('./VGG_ShoeV2_model_best.pth', map_location=device))

    with torch.no_grad():
        model.eval()
        top1_eval, top10_eval = Evaluate_FGSBIR(model, dataloader_Test)
        print(top1_eval, top10_eval)
