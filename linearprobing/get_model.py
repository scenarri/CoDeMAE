import torch
import os
import model.DeCUR as DeCUR
import model.CROMA as CROMA
import model.SatVit as SatVit
import model.SwinSSL.SwinSSL as Swin
import model.DINOMM.DINOMM as DINOMM
import model.fg_mae as fg_mae
from utils import seed_everything
import model.ssl4eo_s12_model as ssl4eo_s12_model
import model.MARS as MARS
import model.itpn as itpn
import model.DOFA as DOFA

def get_model(args):
    if args.backbone == 'DeCUR':
        model = DeCUR.DeCUR(modality=args.modality, num_classes=args.num_classes)
    
    elif args.backbone == 'SS4EOS12':
        model = ssl4eo_s12_model.SS4EOS12(modality=args.modality, num_classes=args.num_classes)
        
    elif args.backbone == 'SatViT':
        model = SatVit.SatViT_Model(modality=args.modality, num_classes=args.num_classes)
        
    elif args.backbone == 'SwinSSL':
        model = Swin.SwinSSL(modality=args.modality, num_classes=args.num_classes)
        
    elif args.backbone == 'FGMAE':
        model = fg_mae.FGMAE(modality=args.modality, num_classes=args.num_classes)
        
    elif args.backbone == 'DINOMM':
        model = DINOMM.dinomm(modality=args.modality, num_classes=args.num_classes)
            
    elif args.backbone == 'CROMA':
        model = CROMA.PretrainedCROMA(modality=args.modality, num_classes=args.num_classes, image_resolution=args.input_size)
        
    elif args.backbone == 'MARS':
        model = MARS.MARS(modality=args.modality, num_classes=args.num_classes)
    
    elif args.backbone == 'CoDeMAE':
        model = itpn.itpn(modality=args.modality, num_classes=args.num_classes)
        
    elif args.backbone == 'DOFA':
        model = DOFA.DOFA(modality=args.modality, num_classes=args.num_classes, dataset=args.dataset)
 
    
    seed_everything(args.seed)
    
    model.head.weight.data.normal_(mean=0.0, std=0.01)
    model.head.bias.data.zero_()
    bn = torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6)
    
    if args.linprob:
        model.head = torch.nn.Sequential(bn, model.head)
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
        if args.modality == 'Both':
            for _, p in model.fuse.named_parameters():
                p.requires_grad = True
        
    return model






