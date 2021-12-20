import argparse, os, torch
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
import warnings
warnings.filterwarnings('ignore')
from mydataloader import CustomImageDataset
import torchvision.transforms as transforms#转换图片
from Model.myModel import Mymodel
from Trainer.mytrain import Trainer
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--checkpointPath', type=str, default = './checkpoints/id_0.pth.tar')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--test_videoLen', type=int, default=3)
def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(1)    
    args = parser.parse_args()
    root="E:\\jianguoyun\\instruments18_caption\\"
    annotations_path='E:\\jianguoyun\\myTripletNUS\\data'
    train_txt_dir='E:\\jianguoyun\\myTripletNUS\\data\\train.txt'
    map_dir='E:\\jianguoyun\\myTripletNUS\\data\\map.txt'
    instrument_dir='E:\\jianguoyun\\myTripletNUS\\data\\instrument.txt'
    verb_dir='E:\\jianguoyun\\myTripletNUS\\data\\verb.txt'

    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    traindata=CustomImageDataset(annotations_path,root,transforms_train,train_txt_dir,map_dir,instrument_dir,verb_dir)

    batch_size = 12
    train_data = DataLoader(traindata, batch_size=batch_size, shuffle=True,drop_last=False)

    test_txt_dir='E:\\jianguoyun\\myTripletNUS\\data\\test.txt'
    transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testdata=CustomImageDataset(annotations_path,root,transforms_test,test_txt_dir,map_dir,instrument_dir,verb_dir)
    test_data = DataLoader(testdata, batch_size=batch_size)
    ngt=32
    num_classes=[9,13,2,32]
    hidden_size=5120
    loss_weight=[1,1,1]



    model=Mymodel(ngt, num_classes, hidden_size,loss_weight)

    model=model.to(device)
    #model = nn.DataParallel(model)
    trainer=Trainer(args,model,train_data,test_data,batch_size)
    trainer.train_model()

#python3.7 -m main --bert_directory bert-base-uncased --num_generated_triples 15 --max_grad_norm 2.5 --na_rel_coef 0.25 --max_epoch 100 --max_span_length 10
