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
from tqdm import tqdm
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,hamming_loss
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='manual to this script')
#parser.add_argument('--checkpointPath', type=str, default = './checkpoints/27_final.pth.tar')
#parser.add_argument('--resume', type=bool, default=False)
#parser.add_argument('--max_epoch', type=int, default=100)
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
def calculate_metrics(pred, target, threshold=0.8):
    #print(pred.shape)
    
    
    pred = np.array(pred > threshold, dtype=float)
    l=hamming_loss(target,pred)
    idx = np.argwhere(np.all(target[..., :] == 0, axis=0))
    target = np.delete(target, idx, axis=1)
    pred=np.delete(pred, idx, axis=1)

    
    
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            'hamming_loss':l
            }

def _compute_AP(gt_labels, pd_probs, valid=None):
    """ Compute the average precision (AP) of each of the 100 considered triplets.
        Args:
            gt_labels: 1D (batch of) vector[N] of integer values 0's or 1's for the groundtruth labels.
            pd_probs:  1D (batch of) vector[N] of float values [0,1] for the predicted labels.
        Returns:
            results:   1D vector[N] of AP for each class 
    """
    gt_instances  = np.sum(gt_labels, axis=0)
    pd_instances  = np.sum(pd_probs, axis=0)
    computed_ap   = average_precision_score(gt_labels, pd_probs, average=None)
    actual_ap     = []
    num_classes   = np.shape(gt_labels)[-1]
    for k in range(num_classes):
        if ((gt_instances[k] != 0) or (pd_instances[k] != 0)) and not np.isnan(computed_ap[k]):
            actual_ap.append(computed_ap[k])
        else:
            actual_ap.append("n/a")
    return actual_ap
def _average_by_videos(results):
    """ Compute the average AP of each triplet class across all the videos
        and mean AP of the model on the triplet predictions.
        Args:
            results:   1D (batch of) vector of AP for each class. One member of the batch corresponds
                       to one video
        Returns:
            AP:   1D vector[N] of AP for each class averaged by videos
    """
    n = results.shape[-1]
    AP = []
    for j in range(n):
        x = results[:,j]
        x = np.mean([float(a) for a in x if (str(a)!='n/a') ])
        if np.isnan(x):
            AP.append("n/a")
        else:          
            AP.append(x)
    mAP = np.mean( [i for i in AP if i !='n/a'])
    return np.array(AP), mAP


if __name__ == '__main__':
    set_seed(1)    
    args = parser.parse_args()
    root="E:\\jianguoyun\\instruments18_caption\\"
    annotations_path='E:\\jianguoyun\\myTripletNUS\\data'
    train_txt_dir='E:\\jianguoyun\\myTripletNUS\\data\\train.txt'
    map_dir='E:\\jianguoyun\\myTripletNUS\\data\\map.txt'
    instrument_dir='E:\\jianguoyun\\myTripletNUS\\data\\instrument.txt'
    verb_dir='E:\\jianguoyun\\myTripletNUS\\data\\verb.txt'

    # transforms_train = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # traindata=CustomImageDataset(annotations_path,root,transforms_train,train_txt_dir,map_dir,instrument_dir,verb_dir)

    batch_size = 12
    # train_data = DataLoader(traindata, batch_size=batch_size, shuffle=True,drop_last=False)

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
    checkpoint = torch.load('E:\\jianguoyun\\n32\\Abstudy\\BCEloss\\Absstudy\\our\\checkpoints\\best\\bestmAP.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    lee_dic={}
    for idx  in range(args.test_videoLen):
        if not idx in lee_dic.keys():
            lee_dic[idx] = {'GT':[], 'PR':[],'IGT':[], 'IPR':[],'VGT':[], 'VPR':[],'TGT':[], 'TPR':[]}
        
    with torch.no_grad():
        model_result = []
        targets = []
       # avg_loss = AverageMeter()
        ap_vid=[]
        iap_vid=[]
        vap_vid=[]
        tap_vid=[]
        
        for data in tqdm(test_data):

            img,triplet,instrumentList,verbList,targetList,instrumentGT,verbGT,targetGT,mask,video_id=data
            img=img.to(device)
            GT=triplet.to(device)
            instrumentList=instrumentList.to(device)
            verbList=verbList.to(device)
            targetList=targetList.to(device)
            instrumentGT=instrumentGT.to(device)
            verbGT=verbGT.to(device)
            targetGT=targetGT.to(device)
            mask=mask.to(device)
            _,outputs= model(img,GT,instrumentList,verbList,targetList,instrumentGT,verbGT,targetGT,mask)
            #loss,outputs= model(img,smooth_one_hot(GT),instrumentList,verbList,targetList,smooth_one_hot(instrumentGT),smooth_one_hot(verbGT),smooth_one_hot(targetGT),mask)
           # avg_loss.update(loss.item(),batch_size)
            pred_head, pred_rel,pred_tail,PR=outputs
           # print(pred_head.shape)#12,32,9
            pred_head=torch.softmax(pred_head,dim=2)
            pred_head,_=torch.max(pred_head,1)
            pred_rel=torch.softmax(pred_rel,dim=2)
            pred_rel,_=torch.max(pred_rel,1)
            pred_tail=torch.softmax(pred_tail,dim=2)
            pred_tail,_=torch.max(pred_tail,1)
            #print(pred_head.shape)#12,9
            pred_head=pred_head[:,:8]
            #print(pred_head.shape)#12,9
            #print(instrumentGT)
           # aaaa
            #max1,_=torch.max(PR,dim=1)
            #max1=max1.unsqueeze(1).repeat(1,PR.shape[1])
            PR=torch.sigmoid(PR)
            gt=GT.cpu().numpy()
            pr=PR.cpu().numpy()
            gt.dtype='int64'
            pr.dtype='float32'
            igt=instrumentGT.cpu().numpy()
            vgt=verbGT.cpu().numpy()
            tgt=targetGT.cpu().numpy()
            #igt.dtype='int64'
            #print(igt)
            #aaaa
            ipr=pred_head.cpu().numpy()
            ipr.dtype='float32'

            vpr=pred_rel.cpu().numpy()
            vpr.dtype='float32'

            tpr=pred_tail.cpu().numpy()
            tpr.dtype='float32'

            for i in range(int(GT.shape[0])):           
                lee_dic[int(video_id[i].cpu().numpy())]['GT'].append(gt[i])
                lee_dic[int(video_id[i].cpu().numpy())]['PR'].append(pr[i])
                lee_dic[int(video_id[i].cpu().numpy())]['IGT'].append(igt[i])
                lee_dic[int(video_id[i].cpu().numpy())]['IPR'].append(ipr[i])
                lee_dic[int(video_id[i].cpu().numpy())]['VGT'].append(vgt[i])
                lee_dic[int(video_id[i].cpu().numpy())]['VPR'].append(vpr[i])
                lee_dic[int(video_id[i].cpu().numpy())]['TGT'].append(tgt[i])
                lee_dic[int(video_id[i].cpu().numpy())]['TPR'].append(tpr[i])
                
            model_result.extend(PR.cpu().numpy())
            targets.extend(GT.cpu().numpy())
    for idx  in range(args.test_videoLen):
        gt_labels=np.array(lee_dic[idx]['GT'])
        pd_probs=np.array(lee_dic[idx]['PR'])
        ap = _compute_AP(gt_labels=gt_labels, pd_probs=pd_probs)
        ap_vid.append(ap)

        gt_labels=np.array(lee_dic[idx]['IGT'])
        pd_probs=np.array(lee_dic[idx]['IPR'])
        # print(gt_labels.shape)
        # print(pd_probs.shape)
        ap = _compute_AP(gt_labels=gt_labels, pd_probs=pd_probs)
        iap_vid.append(ap)

        gt_labels=np.array(lee_dic[idx]['VGT'])
        pd_probs=np.array(lee_dic[idx]['VPR'])
        ap = _compute_AP(gt_labels=gt_labels, pd_probs=pd_probs)
        vap_vid.append(ap)

        gt_labels=np.array(lee_dic[idx]['TGT'])
        pd_probs=np.array(lee_dic[idx]['TPR'])
        ap = _compute_AP(gt_labels=gt_labels, pd_probs=pd_probs)
        tap_vid.append(ap)
        
    ap_vid=np.array(ap_vid)
    iap_vid=np.array(iap_vid)
    vap_vid=np.array(vap_vid)
    tap_vid=np.array(tap_vid)
    AP, mAP = _average_by_videos(results=ap_vid)
    iAP, imAP = _average_by_videos(results=iap_vid)
    vAP, vmAP = _average_by_videos(results=vap_vid)
    tAP, tmAP = _average_by_videos(results=tap_vid)
    
            
    result = calculate_metrics(np.array(model_result), np.array(targets))       
    print("OP: {:.4f} " "OR: {:.4f} " "OF1: {:.4f}".format(result['micro/precision'], result['micro/recall'], result['micro/f1']))
    print("CP: {:.4f} " "CR: {:.4f} " "CF1: {:.4f}".format(result['macro/precision'], result['macro/recall'], result['macro/f1']))
    #print('test loss:',avg_loss.avg)
    print('hamming_loss:',result['hamming_loss'])
    print('mAP:',mAP)# 0.41375225814649963
    print(AP)
    np.save('E:\\jianguoyun\\n32\\Abstudy\\BCEloss\\Absstudy\\our\\checkpoints\\bestAP.npy',AP)
    print('imAP:',imAP)# 0.6407903891268093
    print(iAP)
    np.save('E:\\jianguoyun\\n32\\Abstudy\\BCEloss\\Absstudy\\our\\checkpoints\\bestiAP.npy',iAP)

    print('vmAP:',vmAP)#0.39893032553766467
    print(vAP)
    np.save('E:\\jianguoyun\\n32\\Abstudy\\BCEloss\\Absstudy\\our\\checkpoints\\bestvAP.npy',vAP)

    print('tmAP:',tmAP)#0.8404976314755621
    print(tAP)
    np.save('E:\\jianguoyun\\n32\\Abstudy\\BCEloss\\Absstudy\\our\\checkpoints\\besttAP.npy',tAP)
    #model = nn.DataParallel(model)
    #trainer=Trainer(args,model,train_data,test_data,batch_size)
    #trainer.train_model()

#python3.7 -m main --bert_directory bert-base-uncased --num_generated_triples 15 --max_grad_norm 2.5 --na_rel_coef 0.25 --max_epoch 100 --max_span_length 10
