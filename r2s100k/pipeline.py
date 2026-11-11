"""R2S100K data adapter utilities for the untouched SSC implementation."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, random, shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
DEFAULT_DATA=Path('/run/media/minhduc/D/Study/r2s100k')
TAXONOMY=ROOT/'taxonomy.py'
SUFFIXES={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def files(folder):
 return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES) if folder.is_dir() else []

def find_gt(root,split,image):
 candidates=[root/f'{split}_labels'/f'{image.name}___fuse.png',root/f'{split}_labels'/'Train-Labels'/f'{image.name}___fuse.png']
 if split=='train': candidates += [root/'train_labels'/'Train-Labels'/f'{image.name}___fuse.png',root/'train_labels_1000'/f'{image.name}___fuse.png']
 return next((p for p in candidates if p.is_file()),None)

def taxonomy():
 spec=importlib.util.spec_from_file_location('ssc_r2s_taxonomy',TAXONOMY)
 if not spec or not spec.loader: raise FileNotFoundError(f'R2S100K taxonomy not found: {TAXONOMY}')
 m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 palette=np.array([m.BACKGROUND_RGB]+[m.CLASSES[i][2] for i in range(1,15)],dtype=np.uint8)
 return m,palette

_PALETTE_LUT=None
def decode(mask,palette):
 global _PALETTE_LUT
 if _PALETTE_LUT is None:
  _PALETTE_LUT=np.full(1<<24,255,dtype=np.uint8)
  keys=(palette[:,0].astype(np.uint32)<<16)|(palette[:,1].astype(np.uint32)<<8)|palette[:,2].astype(np.uint32)
  _PALETTE_LUT[keys]=np.arange(len(palette),dtype=np.uint8)
 rgb=np.asarray(mask.convert('RGB')); lut=_PALETTE_LUT
 packed=(rgb[...,0].astype(np.uint32)<<16)|(rgb[...,1].astype(np.uint32)<<8)|rgb[...,2].astype(np.uint32)
 ids=lut[packed]
 if np.any(ids==255):
  unknown=np.unique(packed[ids==255])[:8]
  raise ValueError(f'Unknown GT palette colors: {[((int(x)>>16)&255,(int(x)>>8)&255,int(x)&255) for x in unknown]}')
 return ids

def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)

def prepare(a):
 _,palette=taxonomy(); root=a.dataset_root.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
 stage=out/'.prepare_staging'
 if stage.exists(): shutil.rmtree(stage)
 stage.mkdir(parents=True)
 print(f'[prepare] scanning dataset: {root}',flush=True)
 splits={s:files(root/s) for s in ('train','val','test')}
 print('[prepare] image counts: '+', '.join(f'{s}={len(v)}' for s,v in splits.items()),flush=True)
 print('[prepare] hashing val/test images to detect train leakage...',flush=True)
 held_names={p.name for s in ('val','test') for p in splits[s]}; held_hashes={digest(p) for s in ('val','test') for p in splits[s]}
 print(f'[prepare] held-out hashes ready ({len(held_hashes)} unique); auditing images and GT...',flush=True)
 report={'dataset_root':str(root),'counts':{s:len(v) for s,v in splits.items()},'missing_gt':{},'excluded_train':[],'bad_dimensions':[],'train_label_source':'GT only to derive image-level labels; pixel GT is held for evaluation'}
 for split,images in splits.items():
  print(f'[prepare] {split}: auditing {len(images)} images',flush=True)
  rows=[]; labels=[]; seen=set(); missing=[]
  for index,im in enumerate(images,1):
   h=digest(im)
   if split=='train' and (im.name in held_names or h in held_hashes or h in seen):
    report['excluded_train'].append({'image_id':im.name,'sha256':h}); continue
   seen.add(h); gt=find_gt(root,split,im)
   rec={'image_id':im.name,'artifact_id':hashlib.sha1(im.name.encode()).hexdigest()[:16],'image_path':str(im.resolve())}
   if gt:
    with Image.open(im) as image, Image.open(gt) as mask:
     if image.size!=mask.size: report['bad_dimensions'].append(im.name); continue
     ids=decode(mask,palette)
    present=np.unique(ids)
    rec['image_labels']=[int(x in present) for x in range(1,15)] if split=='train' else None
    labels.append({'image_id':im.name,'gt_path':str(gt.resolve())})
   else:
    missing.append(im.name)
    if split=='train': raise FileNotFoundError(f'Missing train GT for {im.name}; expected under train_labels/Train-Labels')
   rows.append(rec)
   if index%250==0 or index==len(images):
    print(f'[prepare] {split}: {index}/{len(images)} checked',flush=True)
  atomic_json(stage/f'{split}_images.json',rows); atomic_json(stage/f'{split}_labels.json',labels)
  report['missing_gt'][split]=missing; report.setdefault('counts_after_filter',{})[split]=len(rows)
 if report['bad_dimensions']: raise ValueError(f'Image/GT dimension mismatches: {report["bad_dimensions"][:8]}')
 if report['missing_gt'].get('train'): raise FileNotFoundError(f"Missing train GT for {len(report['missing_gt']['train'])} images; see audit")
 train_rows=json.loads((stage/'train_images.json').read_text())
 rng=random.Random(3223); chosen=list(train_rows); rng.shuffle(chosen)
 atomic_json(stage/'smoke32_images.json',chosen[:min(32,len(chosen))]); atomic_json(stage/'pilot1000_images.json',chosen[:min(1000,len(chosen))])
 atomic_json(stage/'audit_report.json',report)
 print('[prepare] audit passed; publishing manifests...',flush=True)
 for item in stage.iterdir(): item.replace(out/item.name)
 stage.rmdir()
 print(f'[prepare] complete: manifests and audit report are in {out}',flush=True)

def superpixels(a):
 try: from skimage.segmentation import slic
 except ImportError as e: raise RuntimeError('Install scikit-image in SSC environment') from e
 from skimage.util import img_as_float
 manifest=json.loads(a.manifest.read_text()); out=a.output; out.mkdir(parents=True,exist_ok=True)
 print(f'[superpixels] start: samples={len(manifest)} segments={a.segments} compactness={a.compactness} sigma={a.sigma} output={out}',flush=True)
 started=time.monotonic(); created=0; skipped=0
 for n,r in enumerate(manifest,1):
  dst=out/f"{r.get('artifact_id',hashlib.sha1(r['image_id'].encode()).hexdigest()[:16])}.npy"
  if dst.exists() and not a.overwrite: skipped+=1
  else:
   with Image.open(r['image_path']) as im: rgb=np.asarray(im.convert('RGB'))
   labels=slic(img_as_float(rgb),n_segments=a.segments,compactness=a.compactness,sigma=a.sigma,start_label=0,channel_axis=-1,convert2lab=True)
   _,labels=np.unique(labels,return_inverse=True)
   if labels.max(initial=0)>np.iinfo(np.uint16).max: raise ValueError(f'SLIC generated too many regions for uint16: {r["image_id"]}')
   np.save(dst,labels.reshape(rgb.shape[:2]).astype(np.uint16)); created+=1
  done=n
  interval=max(1,len(manifest)//20)
  if n==1 or n%interval==0 or n==len(manifest):
   elapsed=time.monotonic()-started; eta=(len(manifest)-n)/(n/max(elapsed,1e-6))
   print(f'[superpixels] {n}/{len(manifest)} ({100*n/max(len(manifest),1):.1f}%) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m image={r["image_id"]}',flush=True)
 print(f'[superpixels] complete rows={done} created={created} reused={skipped} output={out}',flush=True)

def evaluate(a):
 _,palette=taxonomy(); records=json.loads(a.manifest.read_text()); gt_records=json.loads(a.labels.read_text()); gtd={r['image_id']:Path(r['gt_path']) for r in gt_records}
 cm=np.zeros((15,15),np.int64); out=a.output; out.mkdir(parents=True,exist_ok=True)
 print(f'[evaluate] start: samples={len(records)} classes=15 predictions={a.predictions} output={out}',flush=True); started=time.monotonic()
 for i,r in enumerate(records,1):
  image_id=r['image_id']; pred_path=a.predictions/f"{r.get('artifact_id',hashlib.sha1(image_id.encode()).hexdigest()[:16])}.png"
  if image_id not in gtd or not pred_path.is_file(): raise FileNotFoundError(f'Missing prediction or GT for {image_id}: {pred_path}')
  with Image.open(gtd[image_id]) as g, Image.open(pred_path) as p: gt=decode(g,palette); pr=np.asarray(p.convert('L'))
  if gt.shape!=pr.shape: raise ValueError(f'Size mismatch for {image_id}: GT {gt.shape}, prediction {pr.shape}')
  if pr.size and pr.max()>14: raise ValueError(f'Prediction has class ID >14: {pred_path}')
  cm+=np.bincount((gt.astype(np.int64)*15+pr.astype(np.int64)).ravel(),minlength=225).reshape(15,15)
  interval=max(1,len(records)//20)
  if i==1 or i%interval==0 or i==len(records):
   elapsed=time.monotonic()-started; eta=(len(records)-i)/(i/max(elapsed,1e-6))
   print(f'[evaluate] {i}/{len(records)} ({100*i/max(len(records),1):.1f}%) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m image={image_id}',flush=True)
 tp=cm.diagonal(); gt_count=cm.sum(1); pred_count=cm.sum(0); union=pred_count+gt_count-tp; denom=pred_count+gt_count
 iou=np.divide(tp,union,out=np.full(15,np.nan),where=union>0); dice=np.divide(2*tp,denom,out=np.full(15,np.nan),where=denom>0)
 precision=np.divide(tp,pred_count,out=np.full(15,np.nan),where=pred_count>0); recall=np.divide(tp,gt_count,out=np.full(15,np.nan),where=gt_count>0)
 def mean(x):
  x=x[np.isfinite(x)]; return float(x.mean()) if len(x) else None
 fg_tp=int(cm[1:,1:].sum()); fg_fp=int(cm[0,1:].sum()); fg_fn=int(cm[1:,0].sum()); fg_union=fg_tp+fg_fp+fg_fn
 metrics={'evaluated_images':len(records),'miou_15':mean(iou),'miou_foreground14':mean(iou[1:]),'foreground_binary_iou':fg_tp/fg_union if fg_union else None,'foreground_binary_dice':2*fg_tp/(2*fg_tp+fg_fp+fg_fn) if 2*fg_tp+fg_fp+fg_fn else None,'class_metrics':{str(k):{'iou':None if not np.isfinite(iou[k]) else float(iou[k]),'dice':None if not np.isfinite(dice[k]) else float(dice[k]),'precision':None if not np.isfinite(precision[k]) else float(precision[k]),'recall':None if not np.isfinite(recall[k]) else float(recall[k]),'gt_pixels':int(cm[k].sum()),'pred_pixels':int(cm[:,k].sum())} for k in range(15)}}
 atomic_json(out/'metrics.json',metrics); np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
 with open(out/'class_metrics.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['class_id','name','iou','dice','precision','recall','gt_pixels','pred_pixels']); w.writerows([[k,'Background' if k==0 else taxonomy()[0].CLASSES[k][0],metrics['class_metrics'][str(k)]['iou'],metrics['class_metrics'][str(k)]['dice'],metrics['class_metrics'][str(k)]['precision'],metrics['class_metrics'][str(k)]['recall'],int(cm[k].sum()),int(cm[:,k].sum())] for k in range(15)])
 print(f'[evaluate] complete mIoU15={metrics["miou_15"]} mIoU14={metrics["miou_foreground14"]} metrics={out/"metrics.json"}',flush=True)

def smoke(a):
 """Run every stage on a small deterministic train subset to validate wiring."""
 from argparse import Namespace
 from adapter import train_cam, make_cam, make_cam_masks, make_ir_labels, train_irn, make_pseudo
 run=a.output.resolve(); run.mkdir(parents=True,exist_ok=True)
 prepared=a.prepared.resolve()
 source=prepared/'smoke32_images.json'
 if not source.is_file(): raise FileNotFoundError(f'Missing {source}; run prepare first')
 rows=json.loads(source.read_text())[:a.count]
 if not rows: raise ValueError('Smoke manifest is empty')
 manifest=run/'smoke_images.json'; atomic_json(manifest,rows)
 print(f'[smoke] starting end-to-end validation: samples={len(rows)} run={run} device={a.device}',flush=True)
 sp=run/'superpixels'; cam_ckpt=run/'cam_checkpoint.pt'; cams=run/'cams'; cam_masks=run/'cam_masks'; labels=run/'ir_labels'; irn_ckpt=run/'irn_checkpoint.pt'; pseudo=run/'pseudo_masks'; scores=run/'scores'
 superpixels(Namespace(manifest=manifest,output=sp,segments=400,compactness=10.0,sigma=1.0,overwrite=False))
 ids=[r.get('artifact_id') or hashlib.sha1(r['image_id'].encode()).hexdigest()[:16] for r in rows]
 def reusable(folder,suffix): return a.reuse_existing and all((folder/f'{aid}{suffix}').is_file() for aid in ids)
 if a.reuse_existing and cam_ckpt.is_file(): print('[smoke] reusing CAM checkpoint',flush=True)
 else: train_cam(Namespace(manifest=manifest,superpixels=sp,output=cam_ckpt,device=a.device,workers=0,batch_size=1,crop=512,epochs=3,lr=.1/6,resume=False))
 if reusable(cams,'.npz'): print('[smoke] reusing CAM maps',flush=True)
 else: make_cam(Namespace(manifest=manifest,checkpoint=cam_ckpt,output=cams,device=a.device,long_side=256,scales=[1.0,.5],class_mode='manifest',class_threshold=0.0))
 if reusable(cam_masks,'.png'): print('[smoke] reusing CAM masks',flush=True)
 else: make_cam_masks(Namespace(manifest=manifest,cams=cams,output=cam_masks,bg_threshold=.25))
 if reusable(labels,'.png'): print('[smoke] reusing affinity labels',flush=True)
 else: make_ir_labels(Namespace(manifest=manifest,cams=cams,output=labels,fg_threshold=.30,bg_threshold=.10,device=a.device))
 if a.reuse_existing and irn_ckpt.is_file(): print('[smoke] reusing IRNet checkpoint',flush=True)
 else: train_irn(Namespace(manifest=manifest,labels=labels,output=irn_ckpt,device=a.device,workers=0,batch_size=1,crop=512,epochs=1,lr=.1/16,weight_decay=1e-4,resume=False))
 if reusable(pseudo,'.png') and reusable(scores,'.npz'):
  print('[smoke] reusing pseudo masks/scores',flush=True)
 else: make_pseudo(Namespace(manifest=manifest,cams=cams,checkpoint=irn_ckpt,output=pseudo,scores=scores,device=a.device,long_side=256,crop=512,beta=10.0,exp_times=8,bg_threshold=.25))
 eval_labels=prepared/'train_labels.json'
 evaluate(Namespace(manifest=manifest,labels=eval_labels,predictions=cam_masks,output=run/'evaluation_cam'))
 evaluate(Namespace(manifest=manifest,labels=eval_labels,predictions=pseudo,output=run/'evaluation'))
 test_source=prepared/'test_images.json'; test_gt=prepared/'test_labels.json'
 if test_source.is_file() and test_gt.is_file():
  test_rows=json.loads(test_source.read_text())[:a.count]
  test_manifest=run/'smoke_test_images.json'; atomic_json(test_manifest,test_rows)
  test_cams=run/'test_cams'; test_cam_masks=run/'test_cam_masks'; test_pseudo=run/'test_pseudo_masks'; test_scores=run/'test_scores'
  make_cam(Namespace(manifest=test_manifest,checkpoint=cam_ckpt,output=test_cams,device=a.device,long_side=256,scales=[1.0,.5],class_mode='predicted',class_threshold=0.0))
  make_cam_masks(Namespace(manifest=test_manifest,cams=test_cams,output=test_cam_masks,bg_threshold=.25))
  make_pseudo(Namespace(manifest=test_manifest,cams=test_cams,checkpoint=irn_ckpt,output=test_pseudo,scores=test_scores,device=a.device,long_side=256,crop=512,beta=10.0,exp_times=8,bg_threshold=.25))
  evaluate(Namespace(manifest=test_manifest,labels=test_gt,predictions=test_cam_masks,output=run/'evaluation_test_cam'))
  evaluate(Namespace(manifest=test_manifest,labels=test_gt,predictions=test_pseudo,output=run/'evaluation_test_pseudo'))
 print(f'[smoke] all stages passed; outputs={run}',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('prepare'); q.add_argument('--dataset-root',type=Path,default=DEFAULT_DATA); q.add_argument('--output',type=Path,default=ROOT/'runs'/'r2s100k'); q.set_defaults(func=prepare)
 q=sub.add_parser('superpixels'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.add_argument('--segments',type=int,default=400); q.add_argument('--compactness',type=float,default=10); q.add_argument('--sigma',type=float,default=1); q.add_argument('--overwrite',action='store_true'); q.set_defaults(func=superpixels)
 q=sub.add_parser('evaluate'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--labels',type=Path,required=True); q.add_argument('--predictions',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.set_defaults(func=evaluate)
 q=sub.add_parser('smoke',help='Run all stages on a small subset to validate the end-to-end flow'); q.add_argument('--prepared',type=Path,default=ROOT/'runs'/'data'); q.add_argument('--output',type=Path,required=True); q.add_argument('--count',type=int,default=4); q.add_argument('--device',default='cuda'); q.add_argument('--reuse-existing',action='store_true',help='Reuse complete smoke artifacts to resume after fixing a later-stage bug'); q.set_defaults(func=smoke)
 q=sub.add_parser('train-cam'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--superpixels',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.add_argument('--device',default='cuda'); q.add_argument('--workers',type=int,default=4); q.add_argument('--batch-size',type=int,default=6); q.add_argument('--crop',type=int,default=512); q.add_argument('--epochs',type=int,default=10); q.add_argument('--lr',type=float,default=.1); q.add_argument('--resume',action='store_true'); q.set_defaults(func=lambda a: __import__('adapter').train_cam(a))
 q=sub.add_parser('make-cam'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--checkpoint',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.add_argument('--device',default='cuda'); q.add_argument('--long-side',type=int,default=512); q.add_argument('--scales',type=float,nargs='+',default=[1,.5,1.5,2]); q.add_argument('--class-mode',choices=['manifest','predicted'],default='manifest'); q.add_argument('--class-threshold',type=float,default=0.0,help='Classifier logit threshold used with --class-mode predicted'); q.set_defaults(func=lambda a: __import__('adapter').make_cam(a))
 q=sub.add_parser('make-cam-masks'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--cams',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.add_argument('--bg-threshold',type=float,default=.25); q.set_defaults(func=lambda a: __import__('adapter').make_cam_masks(a))
 q=sub.add_parser('make-ir-labels'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--cams',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.add_argument('--fg-threshold',type=float,default=.30); q.add_argument('--bg-threshold',type=float,default=.10); q.add_argument('--device',default='cuda'); q.set_defaults(func=lambda a: __import__('adapter').make_ir_labels(a))
 q=sub.add_parser('train-irn'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--labels',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.add_argument('--device',default='cuda'); q.add_argument('--workers',type=int,default=4); q.add_argument('--batch-size',type=int,default=16); q.add_argument('--crop',type=int,default=512); q.add_argument('--epochs',type=int,default=3); q.add_argument('--lr',type=float,default=.1); q.add_argument('--weight-decay',type=float,default=1e-4); q.add_argument('--resume',action='store_true'); q.set_defaults(func=lambda a: __import__('adapter').train_irn(a))
 q=sub.add_parser('make-pseudo'); q.add_argument('--manifest',type=Path,required=True); q.add_argument('--cams',type=Path,required=True); q.add_argument('--checkpoint',type=Path,required=True); q.add_argument('--output',type=Path,required=True); q.add_argument('--scores',type=Path,required=True); q.add_argument('--device',default='cuda'); q.add_argument('--long-side',type=int,default=512); q.add_argument('--crop',type=int,default=512); q.add_argument('--beta',type=float,default=10); q.add_argument('--exp-times',type=int,default=8); q.add_argument('--bg-threshold',type=float,default=.25); q.set_defaults(func=lambda a: __import__('adapter').make_pseudo(a))
 a=p.parse_args()
 try: a.func(a)
 except Exception as e:
  print(f'[{a.command}] FAILED: {type(e).__name__}: {e}',flush=True)
  raise
if __name__=='__main__': main()
