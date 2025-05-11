
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import util
import test,test_denseuav
import parser
import commons
import datasets_ws
from model import network,network_large,network_t
from model.sync_batchnorm import convert_model
from model.functional import sare_ind, sare_joint
from DenseUAV.tool.utils import load_network

torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

# triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Test set: {test_ds}")

triplets_ds = test_ds
#### Initialize student model
student_model = network.GeoLocalizationNet(args)
student_model = student_model.to(args.device)
if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
    if not args.resume:
        triplets_ds.is_inference = True
        student_model.aggregation.initialize_netvlad_layer(args, triplets_ds, student_model.backbone)
    args.features_dim *= args.netvlad_clusters

student_model = torch.nn.DataParallel(student_model)


#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

if args.criterion == "triplet":
    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")


#### Resume model, optimizer, and other training parameters
if args.resume:
    if args.aggregation != 'crn':
        student_model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train_student(args, student_model, optimizer)
    else:
        # CRN uses pretrained NetVLAD, then requires loading with strict=False and
        # does not load the optimizer from the checkpoint file.
        student_model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train_student(args, student_model, strict=False)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
else:
    best_r5 = start_epoch_num = not_improved_num = 0


if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    student_model = convert_model(student_model)
    student_model = student_model.cuda()


recalls, recalls_str = test_denseuav.test(args, test_ds, student_model, student_model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")



