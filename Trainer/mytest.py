import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
from Trainer.average_meter import AverageMeter
import numpy as np
import os
from torch.optim import lr_scheduler
import gc
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,hamming_loss
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter 
def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def smooth_one_hot(true_labels: torch.Tensor, label_smoothing=0.1):
    smooth_labels = (1.0 - label_smoothing) * true_labels + label_smoothing / true_labels.shape[1]
    return smooth_labels

    

    

def calculate_metrics(pred, target, threshold=0.5):
    #print(pred.shape)
    
    
    pred = np.array(pred > threshold, dtype=float)
    l=hamming_loss(target,pred)
    
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


class Trainer(nn.Module):
    def __init__(self, args,model, train_data,test_data,batch_size):
        super().__init__()
        self.args=args
        self.model = model
        self.train_data = train_data
        self.test_data=test_data
        self.optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),weight_decay=0.001)
        self.batch_size=batch_size
        self.generated_param_directory="./checkpoints/"
        set_seed(1)

    def saveToTxt(self, x, y, filename):
        f = open(filename, 'a+', encoding='utf-8')
        f.write('%d' % x)
        for j in range(len(y)):
            f.write('   %.8f' % y[j])
        f.write('\n')
        f.close()

    def eval_model_new(self):
        self.model.eval()
        lee_dic={}
        for idx  in range(self.args.test_videoLen):
            if not idx in lee_dic.keys():
                lee_dic[idx] = {'GT':[], 'PR':[]}
        # print(self.model.decoder.query_embed.weight)

        with torch.no_grad():
            model_result = []
            targets = []
            avg_loss = AverageMeter()
            ap_vid=[]
          
            for data in tqdm(self.test_data):

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
                loss,outputs= self.model(img,smooth_one_hot(GT),instrumentList,verbList,targetList,smooth_one_hot(instrumentGT),smooth_one_hot(verbGT),smooth_one_hot(targetGT),mask)
                avg_loss.update(loss.item(),self.batch_size)
                PR=outputs[-1]
                #max1,_=torch.max(PR,dim=1)
                #max1=max1.unsqueeze(1).repeat(1,PR.shape[1])
                PR=torch.sigmoid(PR)
                gt=GT.cpu().numpy()
                pr=PR.cpu().numpy()
                gt.dtype='int64'
                pr.dtype='float32'
                for i in range(int(GT.shape[0])):           
                    lee_dic[int(video_id[i].cpu().numpy())]['GT'].append(gt[i])
                    lee_dic[int(video_id[i].cpu().numpy())]['PR'].append(pr[i])
                model_result.extend(PR.cpu().numpy())
                targets.extend(GT.cpu().numpy())
        for idx  in range(self.args.test_videoLen):
            gt_labels=np.array(lee_dic[idx]['GT'])
            pd_probs=np.array(lee_dic[idx]['PR'])
            ap = _compute_AP(gt_labels=gt_labels, pd_probs=pd_probs)
            ap_vid.append(ap)
          
        ap_vid=np.array(ap_vid)
        AP, mAP = _average_by_videos(results=ap_vid)
                
        result = calculate_metrics(np.array(model_result), np.array(targets))       
        print("micro precision: {:.8f} " "micro recall: {:.8f} " "micro f1: {:.8f}".format(result['micro/precision'], result['micro/recall'], result['micro/f1']))
        print('test loss:',avg_loss.avg)
        print('hamming_loss:',result['hamming_loss'])
        print('mAP:',mAP)
        return result,avg_loss.avg,mAP
    def train_model(self):
        writer = SummaryWriter('./log')
        best_f1 =0
        #scheduler = lr_scheduler.StepLR(self.optimizer, 30, gamma=0.1, last_epoch=-1)
        start_epoch=-1
        if self.args.resume and os.path.exists(self.args.checkpointPath):
            print('resume from checkpoint!')
            checkpoint = torch.load(self.args.checkpointPath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            start_epoch = checkpoint['epoch']
            #best_f1 =checkpoint['best_result']

            print('start_epoch:', start_epoch)
            #torch.cuda.empty_cache()
        if start_epoch==-1:
            start_epoch=0
        #lr_list = []

        for epoch in range(start_epoch):

            if epoch % 30 == 0 and epoch!=0:
                for p in self.optimizer.param_groups:
                    p['lr'] *= 0.9
        #lr_list.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
        for epoch in range(start_epoch,self.args.max_epoch):
            # Train
            if epoch % 30 == 0 and epoch!=0:
                for p in self.optimizer.param_groups:
                    p['lr'] *= 0.9
           # lr_list.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            
            self.model.train()
            self.model.zero_grad()
            train_loader=self.train_data
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            batch_id=0
            count=0
            for data in tqdm(train_loader):
                #break
                start = batch_id * self.batch_size
                #image,_,instrument,tripletList,mask=data
                img,triplet,instrumentList,verbList,targetList,instrumentGT,verbGT,targetGT,mask,_=data
                img=img.to(device)
                triplet=triplet.to(device)
                instrumentList=instrumentList.to(device)
                verbList=verbList.to(device)
                targetList=targetList.to(device)
                instrumentGT=instrumentGT.to(device)
                verbGT=verbGT.to(device)
                targetGT=targetGT.to(device)
                
                instrumentGT=smooth_one_hot(instrumentGT)
                verbGT=smooth_one_hot(verbGT)
                targetGT=smooth_one_hot(targetGT)
                triplet=smooth_one_hot(triplet)
                mask=mask.to(device)
                loss,_= self.model(img,triplet,instrumentList,verbList,targetList,instrumentGT,verbGT,targetGT,mask)
                avg_loss.update(loss.item(), self.batch_size)

                #torch.cuda.empty_cache()
                # Optimize
                loss.backward()
                #torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                self.optimizer.step()
                self.model.zero_grad()
                # if batch_id>50:
                #     break
                if batch_id % 100 == 0 and batch_id != 0:
                    tqdm.write("     Instance: %d; loss: %.8f with lr: %.8f" % (start, avg_loss.avg,self.optimizer.state_dict()['param_groups'][0]['lr']))
                    #break
                    checkpoint_dict = {'epoch': epoch,
                                       'model_state_dict': self.model.state_dict(),
                                       'best_result':best_f1
                                       #'optim_state_dict': self.optimizer.state_dict()
                                       }

                    torch.save(checkpoint_dict, './checkpoints/id_%d.pth.tar' % (int(count % 2)))
                    count+=1
                batch_id+=1

            gc.collect()
            torch.cuda.empty_cache()
            print(epoch,avg_loss.avg)
            writer.add_scalar('train_loss', avg_loss.avg,epoch)
            checkpoint_dict = {'epoch': epoch,
                               'model_state_dict': self.model.state_dict(),
                               'best_result':best_f1
                               #'optim_state_dict': self.optimizer.state_dict()
                               }

            torch.save(checkpoint_dict, './checkpoints/%d_final.pth.tar' % (epoch))
            self.saveToTxt(int(epoch), [avg_loss.avg],
                           './checkpoints/train_loss.txt')
            #scheduler.step()
            # Test
            print("=== Epoch %d Test ===" % epoch)
            result,testloss,mAP = self.eval_model_new()
            writer.add_scalar('test_mAP', mAP,epoch)
            writer.add_scalar('test_precision', result['micro/precision'],epoch)
            writer.add_scalar('test_recall', result['micro/recall'],epoch)
            writer.add_scalar('test_f1', result['micro/f1'],epoch)
       #     self.saveToTxt(int(epoch), result,
        #                   './checkpoints/test_loss.txt')
            self.saveToTxt(int(epoch), [result['micro/precision'],result['micro/recall'],result['micro/f1'],result['macro/precision'],result['macro/recall'],result['macro/f1'],result['samples/precision'],result['samples/recall'],result['samples/f1'],result['hamming_loss'],testloss,mAP],'./checkpoints/test_loss.txt')
            f1 = mAP
            if f1 > best_f1:
                print("Achieving Best Result on Test Set.")
                print("mAP on Test:",f1)
                checkpoint_dict = {'epoch': epoch,
                                   'model_state_dict': self.model.state_dict(),
                                   'best_result':f1
                                  # 'optim_state_dict': self.optimizer.state_dict()
                                   }
                torch.save(checkpoint_dict,
                           '%s/best/epoch_%d_%.4f.pth.tar' % (self.generated_param_directory, epoch, f1))
                #torch.save({'state_dict': self.model.state_dict()}, self.generated_param_directory + " epoch_%d_f1_%.4f.model" %(epoch, result))
                best_f1 = f1
                best_result_epoch = epoch
            gc.collect()
            torch.cuda.empty_cache()
            torch.save(self.model.state_dict(),'mynet.pth')
        print("Best result on test set is %f achieving at epoch %d." % (best_f1, best_result_epoch))

