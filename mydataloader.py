import torch
import os
from torch.utils.data import Dataset,DataLoader
from path import Path
import torchvision.transforms as T
from imageio import imread
import numpy as np
import random
import cv2
def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CustomImageDataset(Dataset):
    def __init__(self, annotations_path,root, transform,txt_dir,map_dir,instrument_dir,verb_dir,seed=1):
        #np.random.seed(seed)
        #random.seed(seed)
        set_seed(seed)

        self.root = Path(root)
        self.annotations_path=annotations_path
        self.transform = transform
        #print(txt_dir)
        self.scenes = [self.root/folder[:-1]/'left_frames' for folder in open(txt_dir)]
        with open(map_dir) as f:
            self.map = f.readlines()
        with open(instrument_dir) as f:
            self.instrument = f.readlines()
        with open(verb_dir) as f:
            self.verb = f.readlines()
      #  print(self.scenes)
        self.crawl_folders()
    def crawl_folders(self):
        image_set = []
        video_id=0
        for scene in self.scenes:
            #print(self.annotations_path)
            #print(scene.split('\\')[-2].split('_')[-1])
           # aaa
            #print(scene)
            annotations_path=os.path.join(self.annotations_path,scene.split('\\')[-2].split('_')[-1])
            imgs = sorted(scene.files('*.png'))
            i_t=open(os.path.join(annotations_path,'map.txt'))
            triplet=i_t.readlines()

            #for imgName in imgs:
            for i in range(len(imgs)):
                imgName=imgs[i]
                #i=int(imgName.split('/')[-1].split('.')[0][-3:])
                triplet_id=triplet[i].strip().split(' ')
                triplet_id=np.array(triplet_id,np.int64)
                triplet_id=triplet_id[1:]
                samples={'img':imgName,'triplet_id':triplet_id,'video_id':video_id}
                #print(samples,triplet_id.shape)
                image_set.append(samples)
            video_id+=1
        random.shuffle(image_set)
        self.samples = image_set

    def __len__(self):
        return len(self.samples)
       # return len(self.img_labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(sample['img'])
        
        #img=cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        img=cv2.resize(img,(256,256)) 
        
        img=self.transform(img)
        #print('------------')

        triplet=sample['triplet_id']
        
        
        triDic = {'tripletList':[],'instrument': [], 'verb': [], 'target': [],'mask':[]}
        tripletID = (np.array(triplet) == 1).nonzero()[0]

        
        instrumentGT=np.zeros(8)
        verbGT=np.zeros(13)
        targetGT=np.zeros(2)
        

        for id in tripletID:
            tripletMap = self.map[int(id)]
            instrument=tripletMap.split('/')[0]+'\n'
            verb=tripletMap.split('/')[1]+'\n'
                
            p=self.instrument.index(instrument)
            instrumentGT[p]=1
            triDic['instrument'].append(p)

            p1=self.verb.index(verb)
            verbGT[p1]=1
            triDic['verb'].append(p1)

            if tripletMap.split('/')[2].strip()=='kidney':
                    #ThisTarget[0]=1
                triDic['target'].append(0)
                targetGT[0]=1
            else:
                triDic['target'].append(1)
                targetGT[1]=1

            triDic['mask'].append(1)
        while len(triDic['instrument'])<4:
                triDic['instrument'].append(8)
                triDic['verb'].append(12)
                triDic['target'].append(1)
                triDic['mask'].append(0)

        instrumentList=np.array(triDic['instrument'])
        verbList=np.array(triDic['verb'])
        targetList=np.array(triDic['target'])
        mask=np.array(triDic['mask'])
        video_id=sample['video_id']

        return img,triplet,instrumentList,verbList,targetList,instrumentGT,verbGT,targetGT,mask,video_id


if __name__=="__main__":

   # root="/media/mmlab/data_2/lee/instruments18_caption"
    # Data loading code
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),

        T.ToTensor(),
        normalize
    ])
    root="E:\\jianguoyun\\instruments18_caption\\"
    annotations_path='E:\\jianguoyun\\myTripletNUS\\data'
    txt_dir='E:\\jianguoyun\\myTripletNUS\\data\\train.txt'
    map_dir='E:\\jianguoyun\\myTripletNUS\\data\\map.txt'
    instrument_dir='E:\\jianguoyun\\myTripletNUS\\data\\instrument.txt'
    verb_dir='E:\\jianguoyun\\myTripletNUS\\data\\verb.txt'
    testdata=CustomImageDataset(annotations_path,root,train_transform,txt_dir,map_dir,instrument_dir,verb_dir)
    instrumentCount=np.zeros(8)
    verbCount=np.zeros(13)
    targetCount=np.zeros(2)
    tripletCount=np.zeros(32)

    for data in testdata:
        img,triplet,instrumentList,verbList,targetList,instrumentGT,verbGT,targetGT,mask,video_id=data
        instrumentCount+=instrumentGT
        verbCount+=verbGT
        targetCount+=targetGT
        tripletCount+=triplet
    
    print(instrumentCount)#[ 120. 1321.  154.  736.  131.   31. 1093.  141.]
    print(verbCount)
    print(targetCount)
    print(tripletCount)
    np.savetxt('tripletcount.txt',tripletCount)
    np.savetxt('instrumentcount.txt',instrumentCount)
    np.savetxt('verbcount.txt',verbCount)
    np.savetxt('targetcount.txt',targetCount)
        
