"""R2S100K adapters around SSC's original network and propagation code."""
from __future__ import annotations
import json, sys, hashlib
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

HERE=Path(__file__).resolve().parent
SSC_ROOT=HERE.parent
if str(SSC_ROOT) not in sys.path: sys.path.insert(0,str(SSC_ROOT))

def artifact_id(record):
 return record.get('artifact_id') or hashlib.sha1(record['image_id'].encode()).hexdigest()[:16]

def _log_start(stage, rows, **settings):
 details=' '.join(f'{k}={v}' for k,v in settings.items())
 print(f'[{stage}] start: samples={rows} {details}'.rstrip(),flush=True)

def _progress(stage, index, total, started, detail=''):
 interval=max(1,total//20)
 if index==1 or index%interval==0 or index==total:
  elapsed=time.monotonic()-started
  rate=index/max(elapsed,1e-6)
  eta=(total-index)/rate
  suffix=f' {detail}' if detail else ''
  print(f'[{stage}] {index}/{total} ({100*index/max(total,1):.1f}%) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m{suffix}',flush=True)

class R2STrainDataset(Dataset):
 """Image/class-label/SLIC loader; GT masks are intentionally not loaded."""
 def __init__(self,manifest,superpixels,crop=512,training=True):
  self.rows=json.loads(Path(manifest).read_text()); self.sp=Path(superpixels); self.crop=crop; self.training=training
  self.mean=torch.tensor([.485,.456,.406])[:,None,None]; self.std=torch.tensor([.229,.224,.225])[:,None,None]
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; image=np.asarray(Image.open(r['image_path']).convert('RGB')); sid=artifact_id(r)
  sp=np.load(self.sp/f'{sid}.npy').astype(np.int32)
  if sp.shape!=image.shape[:2]: raise ValueError(f'SLIC/image size mismatch: {r["image_id"]}')
  h,w=image.shape[:2]
  if self.training:
   scale=np.random.uniform(.5,1.5); nh=max(self.crop,int(h*scale)); nw=max(self.crop,int(w*scale))
   im=Image.fromarray(image).resize((nw,nh),Image.Resampling.BILINEAR); sm=Image.fromarray(sp.astype(np.int32),mode='I').resize((nw,nh),Image.Resampling.NEAREST)
   image=np.asarray(im); sp=np.asarray(sm,dtype=np.int32)
   y=np.random.randint(0,nh-self.crop+1); x=np.random.randint(0,nw-self.crop+1)
   image=image[y:y+self.crop,x:x+self.crop]; sp=sp[y:y+self.crop,x:x+self.crop]
   if np.random.rand()<.5: image=image[:,::-1].copy(); sp=sp[:,::-1].copy()
  tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255
  tensor=(tensor-self.mean)/self.std
  labels=torch.tensor(r['image_labels'],dtype=torch.float32)
  # Compact labels after crop so one-hot allocation is bounded.
  _,sp=np.unique(sp,return_inverse=True); sp=sp.reshape(image.shape[:2])
  return {'image':tensor,'labels':labels,'superpixel':torch.from_numpy(sp.astype(np.int64)),'image_id':r['image_id']}

class SSC14CAM(nn.Module):
 def __init__(self,pretrained=True):
  super().__init__()
  from net import resnet50_cam
  from net.decoder import Decoder
  from net.model_util import Conv2dBlock, ResBlocks
  self.net=resnet50_cam.Net()
  self.net.classifier=nn.Conv2d(2048,14,1,bias=False)
  # Net.newly_added was constructed with the original VOC classifier; replace
  # that reference too, otherwise SGD silently leaves the R2S head frozen.
  self.net.newly_added=nn.ModuleList([self.net.classifier])
  if pretrained: nn.init.kaiming_normal_(self.net.classifier.weight)
  # Adapt only class dimensionality; decoder blocks and reconstruction path remain SSC's.
  dec=Decoder(); dec.main=nn.Sequential(Conv2dBlock(14,256,3,stride=1,padding=1,norm='ln',activation='relu',pad_type='reflect',bias=False),ResBlocks(1,256,'ln','relu',pad_type='zero'))
  self.decoder=dec
 def forward(self,x):
  net=self.net; z=net.stage1(x); z=net.stage2(z).detach(); z=net.relu(z); z=net.stage3(z); z=net.stage4(z); logits=net.classifier(z)
  return torchutils_gap(logits),logits
 def trainable_parameters(self): return self.net.trainable_parameters()

def torchutils_gap(x):
 # Same global average pooling used by SSC's torchutils.gap2d.
 return F.adaptive_avg_pool2d(x,1).flatten(1)

def alignment_loss(pred,sp,labels):
 # Superpixel-consistency objective used by SSC, implemented without CPU/GPU tensor conversion.
 sp=F.interpolate(sp[:,None].float(),scale_factor=.5,mode='nearest')[:,0].long(); b,h,w=sp.shape
 losses=[]
 for i in range(b):
  active=torch.where(labels[i]>0)[0]
  if active.numel()==0: continue
  p=F.interpolate(pred[i:i+1],size=(h,w),mode='bilinear',align_corners=False)[0,active].flatten(1)
  p=p.clamp_min(0); p=p/(p.amax(dim=1,keepdim=True)+1e-5)
  sid=sp[i].flatten(); n=int(sid.max().item())+1
  sums=torch.zeros((len(active),n),device=p.device).scatter_add_(1,sid[None].expand(len(active),-1),p)
  counts=torch.bincount(sid,minlength=n).clamp_min(1).to(p.dtype)
  avg=sums/counts[None]; avg=avg/(avg.amax(dim=1,keepdim=True)+1e-5)
  original=p.reshape(-1,h,w)
  # SSC suppresses small activation islands with an 8x8 erosion before aligning.
  import cv2
  kernel=np.ones((8,8),np.uint8); maps=original.detach().cpu().numpy(); kept=[]
  for att in maps:
   region=(att>.3).astype(np.uint8); region=cv2.erode(region,kernel,iterations=1)
   kept.append(torch.from_numpy(att*region).to(pred.device))
  eroded=torch.stack(kept).flatten(1)
  target=torch.maximum(avg[:,sid],eroded).detach()
  losses.append(F.mse_loss(p,target))
 return torch.stack(losses).mean() if losses else pred.sum()*0

def train_cam(a):
 device=torch.device(a.device); model=SSC14CAM().to(device); model.decoder.to(device)
 dataset=R2STrainDataset(a.manifest,a.superpixels,crop=a.crop,training=True)
 loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,drop_last=False,pin_memory=device.type=='cuda')
 _log_start('train-cam',len(dataset),device=device,batch=a.batch_size,crop=a.crop,epochs=a.epochs,steps_per_epoch=len(loader),resume=a.resume)
 groups=model.trainable_parameters(); opt=torch.optim.SGD([{'params':groups[0],'lr':a.lr,'weight_decay':1e-4},{'params':groups[1],'lr':10*a.lr,'weight_decay':1e-4}],momentum=.9)
 decopt=torch.optim.Adam(model.decoder.parameters(),lr=.0005,betas=(.5,.999))
 from misc.loss_util import VGGLoss
 vgg=VGGLoss().to(device).eval(); start=0
 if a.resume and Path(a.output).is_file():
  state=torch.load(a.output,map_location=device,weights_only=False); model.load_state_dict(state['model']); model.decoder.load_state_dict(state['decoder']); opt.load_state_dict(state['optimizer']); decopt.load_state_dict(state['decoder_optimizer']); start=state['epoch']
 max_steps=max(1,a.epochs*len(loader))
 for epoch in range(start,a.epochs):
  epoch_started=time.monotonic(); model.train(); model.decoder.train(); total=0
  for step,batch in enumerate(loader):
   rate=a.lr*(1-(epoch*len(loader)+step)/max_steps)**.9
   opt.param_groups[0]['lr']=rate; opt.param_groups[1]['lr']=10*rate
   decopt.param_groups[0]['lr']=.0005*(1-(epoch*len(loader)+step)/max_steps)**.9
   im=batch['image'].to(device); lab=batch['labels'].to(device); sp=batch['superpixel'].to(device)
   logits,cam=model(im); cls=F.multilabel_soft_margin_loss(logits,lab)
   rec=model.decoder(cam); per=vgg(rec,im)
   align=cam.sum()*0 if epoch<2 else alignment_loss(cam,sp,lab)
   loss=cls+per+align; opt.zero_grad(set_to_none=True); decopt.zero_grad(set_to_none=True); loss.backward(); opt.step(); decopt.step(); total+=loss.item()
   if (step+1)==1 or (step+1)%max(1,len(loader)//10)==0 or (step+1)==len(loader):
    print(f'[train-cam] epoch={epoch+1}/{a.epochs} step={step+1}/{len(loader)} loss={loss.item():.5f} cls={cls.item():.5f} perceptual={per.item():.5f} align={align.item():.5f}',flush=True)
  Path(a.output).parent.mkdir(parents=True,exist_ok=True)
  torch.save({'epoch':epoch+1,'model':model.state_dict(),'decoder':model.decoder.state_dict(),'optimizer':opt.state_dict(),'decoder_optimizer':decopt.state_dict()},a.output)
  print(f'[train-cam] epoch {epoch+1}/{a.epochs} complete mean_loss={total/max(1,len(loader)):.5f} elapsed={time.monotonic()-epoch_started:.1f}s checkpoint={a.output}',flush=True)
 print('[train-cam] complete',flush=True)

def _load_cam(path,device):
 from adapter import SSC14CAM
 model=SSC14CAM().to(device); state=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(state['model'] if 'model' in state else state); model.eval(); return model

def _prep_rgb(rgb,size=None):
 im=Image.fromarray(rgb)
 if size:
  w,h=im.size; scale=size/max(w,h); im=im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR)
 arr=np.asarray(im).astype(np.float32)/255
 ten=torch.from_numpy(arr.transpose(2,0,1)).float()
 mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
 return ((ten-mean)/std).unsqueeze(0),im

def _cam_logits(model,x):
 """Run the SSC CAM trunk without its VOC-specific 20-class reshape."""
 net=model.net
 z=net.stage1(x); z=net.stage2(z).detach(); z=net.relu(z); z=net.stage3(z); z=net.stage4(z)
 return net.classifier(z)

@torch.no_grad()
def make_cam(a):
 device=torch.device(a.device); model=_load_cam(a.checkpoint,device); rows=json.loads(Path(a.manifest).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 _log_start('make-cam',len(rows),device=device,scales=a.scales,long_side=a.long_side,class_mode=a.class_mode,class_threshold=a.class_threshold,checkpoint=a.checkpoint)
 started=time.monotonic()
 for ix,r in enumerate(rows,1):
  with Image.open(r['image_path']) as f: rgb=np.asarray(f.convert('RGB'))
  cams=[]
  if a.class_mode=='manifest':
   if r.get('image_labels') is None: raise ValueError(f'{r["image_id"]} has no manifest image labels; use --class-mode predicted for inference without labels')
   keys=np.flatnonzero(np.asarray(r['image_labels'])>0)
  else: keys=None
  for scale in a.scales:
   x,_=_prep_rgb(rgb,max(32,round(a.long_side*scale))); h,w=x.shape[-2:]; x=x.to(device)
   pair=torch.cat([x,x.flip(-1)],0); pred=_cam_logits(model,pair); pred=pred[0]+pred[1].flip(-1)
   if keys is None:
    image_logits=F.adaptive_avg_pool2d(pred[None],1).flatten()
    keys=torch.where(image_logits>a.class_threshold)[0].cpu().numpy()
   pred=pred[keys]
   if len(keys):
    pred=F.relu(pred); pred=pred/(F.adaptive_max_pool2d(pred[None],(1,1))+1e-5)
    pred=F.interpolate(pred,size=(max(1,(h+3)//4),max(1,(w+3)//4)),mode='bilinear',align_corners=False)[0]
    cams.append(pred)
  if len(keys):
   # Accumulate every scale onto a shared 1/4 grid, then normalize per class.
   base=(max(1,(round(rgb.shape[0]*a.long_side/max(rgb.shape[:2]))+3)//4),max(1,(round(rgb.shape[1]*a.long_side/max(rgb.shape[:2]))+3)//4))
   score=torch.stack([F.interpolate(c[None],size=base,mode='bilinear',align_corners=False)[0] for c in cams]).sum(0)
   score=score/(score.flatten(1).amax(1)[:,None,None]+1e-5)
  else:
   base=(max(1,(rgb.shape[0]*a.long_side//max(rgb.shape[:2])+3)//4),max(1,(rgb.shape[1]*a.long_side//max(rgb.shape[:2])+3)//4))
   score=torch.zeros((0,*base),device=device)
  # Store only compact CAMs (about 1/4 of the 512px inference input). The CRF
  # stage upsamples one image at a time, avoiding a full-resolution 14x float map
  # per image on disk (which would cost hundreds of GB on R2S100K).
  np.savez_compressed(out/f'{artifact_id(r)}.npz',keys=keys,cam=score.cpu().numpy().astype(np.float32),image_id=r['image_id'])
  _progress('make-cam',ix,len(rows),started,detail=f'image={r["image_id"]}')
 print(f'[make-cam] complete output={out}',flush=True)

def make_ir_labels(a):
 from misc import imutils
 device=torch.device(a.device); rows=json.loads(Path(a.manifest).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 _log_start('make-ir-labels',len(rows),device=device,fg_threshold=a.fg_threshold,bg_threshold=a.bg_threshold,cams=a.cams)
 started=time.monotonic()
 for ix,r in enumerate(rows,1):
  with Image.open(r['image_path']) as im: rgb=np.asarray(im.convert('RGB'))
  data=np.load(a.cams/f'{artifact_id(r)}.npz'); keys=np.asarray(data['keys'],dtype=np.int64)
  if 'high_res' in data.files: cams=data['high_res']  # read artifacts made by the earlier adapter version
  else:
   compact=torch.from_numpy(data['cam'].astype(np.float32))
   cams=F.interpolate(compact[None],size=rgb.shape[:2],mode='bilinear',align_corners=False)[0].numpy()
  if len(keys)==0: label=np.zeros(rgb.shape[:2],np.uint8)
  else:
   # SSC's CRF helper accepts label maps; keep its foreground/background confidence rules.
   labelset=np.concatenate([np.zeros((1,*rgb.shape[:2]),np.float32),cams],axis=0)
   fg=np.concatenate([np.full((1,*rgb.shape[:2]),a.fg_threshold,np.float32),cams],axis=0).argmax(0)
   bg=np.concatenate([np.full((1,*rgb.shape[:2]),a.bg_threshold,np.float32),cams],axis=0).argmax(0)
   fg=imutils.crf_inference_label(rgb,fg,n_labels=len(keys)+1)
   bg=imutils.crf_inference_label(rgb,bg,n_labels=len(keys)+1)
   semantic=np.r_[0,keys+1]
   fgc=semantic[fg]; bgc=semantic[bg]
   label=fgc.copy(); label[fgc==0]=255; label[(fgc+bgc)==0]=0
  Image.fromarray(label.astype(np.uint8)).save(out/f'{artifact_id(r)}.png')
  _progress('make-ir-labels',ix,len(rows),started,detail=f'image={r["image_id"]}')
 print(f'[make-ir-labels] complete output={out}',flush=True)

@torch.no_grad()
def make_cam_masks(a):
 rows=json.loads(Path(a.manifest).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 _log_start('make-cam-masks',len(rows),background_threshold=a.bg_threshold,cams=a.cams)
 started=time.monotonic()
 for ix,r in enumerate(rows,1):
  with Image.open(r['image_path']) as im: size=(im.height,im.width)
  data=np.load(a.cams/f'{artifact_id(r)}.npz'); keys=np.asarray(data['keys'],dtype=np.int64); compact=data['cam'].astype(np.float32)
  if len(keys):
   maps=F.interpolate(torch.from_numpy(compact)[None],size=size,mode='bilinear',align_corners=False)[0].numpy()
   scores=np.zeros((15,*size),dtype=np.float32); scores[0]=a.bg_threshold; scores[keys+1]=maps
   pred=scores.argmax(0).astype(np.uint8)
  else: pred=np.zeros(size,dtype=np.uint8)
  Image.fromarray(pred).save(out/f'{artifact_id(r)}.png')
  _progress('make-cam-masks',ix,len(rows),started,detail=f'image={r["image_id"]}')
 print(f'[make-cam-masks] complete output={out}',flush=True)

class IRNDataset(Dataset):
 def __init__(self,manifest,labels,crop=512): self.rows=json.loads(Path(manifest).read_text()); self.labels=Path(labels); self.crop=crop
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; aid=artifact_id(r)
  with Image.open(r['image_path']) as f: im=f.convert('RGB')
  with Image.open(self.labels/f'{aid}.png') as f: lab=f.copy()
  w,h=im.size; scale=np.random.uniform(.5,1.5); nw=max(self.crop,round(w*scale)); nh=max(self.crop,round(h*scale))
  im=im.resize((nw,nh),Image.Resampling.BILINEAR); lab=lab.resize((nw,nh),Image.Resampling.NEAREST)
  x=np.random.randint(nw-self.crop+1); y=np.random.randint(nh-self.crop+1)
  im=np.asarray(im)[y:y+self.crop,x:x+self.crop]; lab=np.asarray(lab)[y:y+self.crop,x:x+self.crop]
  if np.random.rand()<.5: im=im[:,::-1].copy(); lab=lab[:,::-1].copy()
  tensor=torch.from_numpy(im.transpose(2,0,1).copy()).float()/255
  tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
  from misc.imutils import pil_rescale
  lab=pil_rescale(lab,.25,0).astype(np.uint8)
  return tensor,torch.from_numpy(lab.astype(np.int64))

def train_irn(a):
 from net import resnet50_irn
 from misc import indexing
 path_index=indexing.PathIndex(radius=10,default_size=(a.crop//4,a.crop//4))
 model=resnet50_irn.AffinityDisplacementLoss(path_index).to(a.device)
 groups=model.trainable_parameters(); opt=torch.optim.SGD([{'params':groups[0],'lr':a.lr,'weight_decay':a.weight_decay},{'params':groups[1],'lr':10*a.lr,'weight_decay':a.weight_decay}],momentum=.9)
 ds=IRNDataset(a.manifest,a.labels,a.crop); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,drop_last=True,pin_memory=True)
 if len(loader)==0: raise ValueError(f'IRNet loader has zero batches: samples={len(ds)} batch_size={a.batch_size}; reduce --batch-size')
 _log_start('train-irn',len(ds),device=a.device,batch=a.batch_size,crop=a.crop,epochs=a.epochs,steps_per_epoch=len(loader),resume=a.resume)
 from voc12.dataloader import GetAffinityLabelFromIndices
 affinity=GetAffinityLabelFromIndices(path_index.src_indices,path_index.dst_indices); start=0
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 if a.resume and out.is_file():
  state=torch.load(out,map_location=a.device,weights_only=False); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); start=state['epoch']
 max_steps=max(1,a.epochs*len(loader))
 for epoch in range(start,a.epochs):
  epoch_started=time.monotonic(); model.train(); running=0
  for step,(im,lab) in enumerate(loader):
   rate=a.lr*(1-(epoch*len(loader)+step)/max_steps)**.9
   opt.param_groups[0]['lr']=rate; opt.param_groups[1]['lr']=10*rate
   bg,fg,neg=zip(*(affinity(x.numpy()) for x in lab)); bg=torch.stack(bg).to(a.device); fg=torch.stack(fg).to(a.device); neg=torch.stack(neg).to(a.device); im=im.to(a.device)
   pos,neg_loss,dpfg,dpbg=model(im,True)
   bg_loss=(bg*pos).sum()/(bg.sum()+1e-5); fg_loss=(fg*pos).sum()/(fg.sum()+1e-5); pos_loss=(bg_loss+fg_loss)/2
   nloss=(neg*neg_loss).sum()/(neg.sum()+1e-5); dfg=(dpfg*fg.unsqueeze(1)).sum()/(2*fg.sum()+1e-5); dbg=(dpbg*bg.unsqueeze(1)).sum()/(2*bg.sum()+1e-5)
   loss=(pos_loss+nloss)/2+(dfg+dbg)/2
   opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); running+=loss.item()
   if (step+1)==1 or (step+1)%max(1,len(loader)//10)==0 or (step+1)==len(loader):
    print(f'[train-irn] epoch={epoch+1}/{a.epochs} step={step+1}/{len(loader)} loss={loss.item():.5f}',flush=True)
  torch.save({'epoch':epoch+1,'model':model.state_dict(),'optimizer':opt.state_dict()},out)
  print(f'[train-irn] epoch {epoch+1}/{a.epochs} complete mean_loss={running/max(1,len(loader)):.5f} elapsed={time.monotonic()-epoch_started:.1f}s checkpoint={out}',flush=True)
 # Estimate displacement mean as SSC does and store final mean in checkpoint.
 model.eval(); vals=[]
 with torch.no_grad():
  for im,_ in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
   _,dp=model(im.to(a.device),False); vals.append(dp.mean((0,2,3)).cpu())
 if vals:
  model.mean_shift.running_mean=torch.stack(vals).mean(0).to(a.device); torch.save({'epoch':a.epochs,'model':model.state_dict(),'optimizer':opt.state_dict()},out)
 print(f'[train-irn] displacement mean estimated; complete checkpoint={out}',flush=True)

@torch.no_grad()
def make_pseudo(a):
 from net import resnet50_irn
 from misc import indexing
 from PIL import Image
 rows=json.loads(Path(a.manifest).read_text()); device=torch.device(a.device)
 irn=resnet50_irn.EdgeDisplacement(crop_size=a.crop).to(device); state=torch.load(a.checkpoint,map_location=device,weights_only=False); irn.load_state_dict(state['model'] if 'model' in state else state,strict=False); irn.eval()
 cams=Path(a.cams); out=Path(a.output); scores=Path(a.scores); out.mkdir(parents=True,exist_ok=True); scores.mkdir(parents=True,exist_ok=True)
 _log_start('make-pseudo',len(rows),device=device,long_side=a.long_side,beta=a.beta,exp_times=a.exp_times,checkpoint=a.checkpoint)
 started=time.monotonic()
 for ix,r in enumerate(rows,1):
  with Image.open(r['image_path']) as f: rgb=np.asarray(f.convert('RGB'))
  x,_=_prep_rgb(rgb,a.long_side); x=x.to(device)
  # SSC's EdgeDisplacement inference forward fuses an image with its horizontal flip.
  pair=torch.cat([x,x.flip(-1)],dim=0); edge,dp=irn(pair)
  d=np.load(cams/f'{artifact_id(r)}.npz'); c=torch.from_numpy(d['cam']).to(device); keys=np.asarray(d['keys'],dtype=np.int64)
  if len(keys)==0:
   pred=np.zeros(rgb.shape[:2],np.uint8); sc=np.zeros((15,*rgb.shape[:2]),np.float32); conf=np.zeros(rgb.shape[:2],np.uint8)
   score_low=np.zeros((15,*edge.shape[-2:]),np.float32)
  else:
   if c.shape[-2:]!=edge.shape[-2:]: c=F.interpolate(c[None],size=edge.shape[-2:],mode='bilinear',align_corners=False)[0]
   rw=indexing.propagate_to_edge(c,edge,beta=a.beta,exp_times=a.exp_times,radius=5)
   if rw.ndim==4: rw=rw[:,0]
   # rw is class x H x W. Normalize class maps, add SSC background prior, then map indices to R2S IDs.
   # SSC normalizes the propagated score tensor by its global maximum.
   rw=rw/(rw.max()+1e-5)
   score_low=np.zeros((15,*rw.shape[-2:]),np.float32); score_low[0]=a.bg_threshold; score_low[keys+1]=rw.cpu().numpy()
   rw=F.interpolate(rw[None],size=rgb.shape[:2],mode='bilinear',align_corners=False)[0]
   sc=torch.zeros((15,*rgb.shape[:2]),device=device); sc[0]=a.bg_threshold; sc[keys+1]=rw
   pred=sc.argmax(0).byte().cpu().numpy(); sc=sc.cpu().numpy()
   margin=np.partition(sc, -2, axis=0)[-1]-np.partition(sc, -2, axis=0)[-2]
   conf=np.zeros(pred.shape,np.uint8)
   for cls in np.unique(pred):
    loc=np.flatnonzero(pred.ravel()==cls); count=max(1,len(loc)//2)
    best=loc[np.argpartition(margin.ravel()[loc],-count)[-count:]]; conf.ravel()[best]=1
  aid=artifact_id(r); Image.fromarray(pred).save(out/f'{aid}.png'); Image.fromarray(conf).save(out/f'{aid}_confidence.png')
  # Scores are auxiliary; keep them at the random-walk feature resolution.
  np.savez_compressed(scores/f'{aid}.npz',score=score_low,image_id=r['image_id'],feature_stride=4,original_shape=np.asarray(pred.shape,dtype=np.int32),inference_long_side=a.long_side)
  _progress('make-pseudo',ix,len(rows),started,detail=f'image={r["image_id"]}')
 print(f'[make-pseudo] complete masks={out} scores={scores}',flush=True)
