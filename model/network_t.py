
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel
from google_drive_downloader import GoogleDriveDownloader as gdd
import math
from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}

class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # out = torch.sign(x)
        p = (x >= 0) * (+1.0)
        n = (x < 0) * (-1.0)
        out = p + n
        # out = x + (torch.rand_like(x)-0.5)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        # x, = ctx.saved_tensors
        # i2 = x.clone().detach()
        # i3 = torch.clamp(i2, -1, 1)
        # mask = (i3 == i2) + 0.0
        return grad_output * 1.0#mask

class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
        self.self_att = False
        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim

    def forward(self, x):
        # x = self.backbone(x)
        # x = self.aggregation(x)
        # return x
        x = self.backbone(x)

        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            # return x.last_hidden_state[:, 0, :]
            B,P,D = x.last_hidden_state.shape
            W = H = int(math.sqrt(P-1))
            x1 = x.last_hidden_state[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2) 
            x = self.aggregation(x1)
        elif self.arch_name.startswith("deit"):
            x = (x.last_hidden_state[:, 0, :]+x.last_hidden_state[:, 1, :])/2.
            return x
        elif self.arch_name.startswith("cct"):
            B,P,D = x.shape
            x = x.view(-1,24,24,384)
            x = x.permute(0, 3, 1, 2) 
            x = self.aggregation(x)
        elif self.arch_name.startswith("dino"):
            # return x["x_norm_clstoken"]
            B,P,D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            x0 = x["x_norm_clstoken"]
            x1 = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 
            # return F.avg_pool2d(x1, kernel_size=W, stride=W).view(B,D)
            x = self.aggregation(x1)
            # x = x["x_norm_clstoken"]
        else:
            x = self.aggregation(x)
        
        # x = self.encoder(x.view(B,-1,D)).view(B,-1)

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x = STE_binary.apply(x)
        return x


def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation in ['cls', 'seqpool']:
        return nn.Identity()


def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        assert args.resize[0] in [224, 384], f'Image size for ViT must be either 224 or 384, but it\'s {args.resize[0]}'
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        backbone = VitWrapper(backbone, args.aggregation)
        
        args.features_dim = 768
        return backbone
    elif args.backbone.startswith("dino"):
        # dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', pretrained=True).eval().cuda()
        # backbone = dino
        from model.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
        backbone = vit_small(patch_size=14,img_size=518,init_values=1,block_chunks=0)

        model_dict = backbone.state_dict()

        state_dict = torch.load("dinov2_vits14_pretrain.pth")
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)

        for p in backbone.parameters():
            p.requires_grad = False
        for name, child in backbone.blocks.named_children():
            if int(name) >= 8:
                for params in child.parameters():
                    params.requires_grad = True

        args.features_dim = 384 #384 #768 #1024
        return backbone

    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone


class VitWrapper(nn.Module):
    def __init__(self, vit_model, aggregation):
        super().__init__()
        self.vit_model = vit_model
        self.aggregation = aggregation
    def forward(self, x):
        if self.aggregation in ["netvlad", "gem"]:
            return self.vit_model(x).last_hidden_state[:, 1:, :]
        else:
            return self.vit_model(x).last_hidden_state[:, 0, :]


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

 