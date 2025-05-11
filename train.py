
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
import test
import parser
import commons
import datasets_ws
from model import network,network_large
from model.sync_batchnorm import convert_model
from model.functional import sare_ind, sare_joint

def calculate_losses_KL(student_features, teacher_features, G, K=5, tau_q=1.0, tau_g=0.01):
    """
    根据论文中的KL散度计算CSD损失：
    LKL = DKL(pg || pq)
    """
    # 计算查询图像与邻居图像的相似性
    C_s = torch.mm(student_features, G.T)  # 学生模型生成的查询图像与图库图像的相似性
    C_t = torch.mm(teacher_features, G.T)  # 教师模型生成的查询图像与图库图像的相似性
    
    # 将相似性转换为概率分布
    p_g = F.softmax(C_t / tau_g, dim=1)  # 教师模型的概率分布，使用温度调节
    p_q = F.softmax(C_s / tau_q, dim=1)  # 学生模型的概率分布，使用温度调节
    
    # 计算KL散度损失
    kl_divergence = F.kl_div(F.log_softmax(C_s / tau_q, dim=1), p_g, reduction='batchmean')  # 计算学生模型与教师模型之间的KL散度

    return kl_divergence

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

triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
logging.info(f"Train query set: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### Initialize student model
student_model = network.GeoLocalizationNet(args)
student_model = student_model.to(args.device)
if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
    if not args.resume:
        triplets_ds.is_inference = True
        student_model.aggregation.initialize_netvlad_layer(args, triplets_ds, student_model.backbone)
    args.features_dim *= args.netvlad_clusters

student_model = torch.nn.DataParallel(student_model)

#### Load teacher model
teacher_model = network_large.GeoLocalizationNet(args)
teacher_model = teacher_model.to(args.device)
if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
    if not args.resume:
        triplets_ds.is_inference = True
        teacher_model.aggregation.initialize_netvlad_layer(args, triplets_ds, teacher_model.backbone)
    args.features_dim *= args.netvlad_clusters

teacher_model = torch.nn.DataParallel(teacher_model)
teacher_model_state_dict = torch.load(join(args.teacher_model_path))["model_state_dict"]
teacher_model.load_state_dict(teacher_model_state_dict)



#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

if args.criterion == "triplet":
    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
elif args.criterion == "sare_ind":
    criterion_triplet = sare_ind
elif args.criterion == "sare_joint":
    criterion_triplet = sare_joint
elif args.criterion == "asymmetric":
    criterion_triplet_1 = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
    criterion_triplet_2 = nn.MSELoss()
else :
    criterion_triplet = calculate_losses_KL

#### Resume model, optimizer, and other training parameters
if args.resume:
    if args.aggregation != 'crn':
        student_model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, student_model, optimizer)
    else:
        # CRN uses pretrained NetVLAD, then requires loading with strict=False and
        # does not load the optimizer from the checkpoint file.
        student_model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, student_model, strict=False)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
else:
    best_r5 = start_epoch_num = not_improved_num = 0

if args.backbone.startswith('vit'):
    logging.info(f"Output dimension of the student_model is {args.features_dim}")
# else:
#     logging.info(f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}")

if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    student_model = convert_model(student_model)
    student_model = student_model.cuda()
    teacher_model = convert_model(teacher_model)
    teacher_model = teacher_model.cuda()

#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    
    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")
        
        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, teacher_model)  # use teacher model to get the positive and negtive samples 
        triplets_ds.is_inference = False
        
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device == "cuda"),
                                 drop_last=True)
        
        student_model = student_model.train()
        
        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
        # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):

            # Flip all triplets or none
            if args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)

            '''
            # reshape images to: train_batch_size*12*3*H*W
            images = images.reshape(args.train_batch_size, 12, 3, images.shape[2], images.shape[3])
            
            # Compute features of all images (images contains queries, positives and negatives)
            # image分三个：query(student)、postive(teacher)、negtive(teacher)
            # 获取查询样本特征
            query_images = images[:, 0, :, :, :]  # 提取每个样本的第一个图像 (查询样本)
            positive_images = images[:, 1:2, :, :, :]  # 提取正样本 (3个正样本中的第一个)
            negative_images = images[:, 2:, :, :, :]  # 提取负样本 (剩下的10个负样本)

            loss_triplet += criterion_triplet(query_features, positive_features, negative_features)

            del query_features,positive_features,negative_features
            '''
            student_features, teacher_features = student_model(images.to(args.device)), teacher_model(images.to(args.device))
            loss_triplet = 0
            
            if args.criterion == "triplet":
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    loss_triplet += criterion_triplet(student_features[queries_indexes],
                                                      teacher_features[positives_indexes],
                                                      teacher_features[negatives_indexes])
            elif args.criterion == "asymmetric":
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    loss_triplet += criterion_triplet_1(student_features[queries_indexes],
                                                      teacher_features[positives_indexes],
                                                      teacher_features[negatives_indexes])

                    loss_triplet += 1024 * criterion_triplet_2(student_features[queries_indexes],
                                                      teacher_features[queries_indexes])
                    
                    loss_triplet += 1024 * criterion_triplet_2(student_features[positives_indexes],
                                                      teacher_features[positives_indexes])
                    
                    loss_triplet += 1024 * criterion_triplet_2(student_features[negatives_indexes],
                                                      teacher_features[negatives_indexes])
            else:
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                # CSD L1/L2
                first_order = True
                alpha = 2
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    if first_order:
                        loss_triplet += torch.norm(torch.matmul(student_features[queries_indexes], teacher_features[queries_indexes].T) 
                                                   - 1,
                                                   p=alpha)
                        
                        loss_triplet += torch.norm(torch.matmul(student_features[queries_indexes], teacher_features[positives_indexes].T) 
                                                   - torch.matmul(teacher_features[queries_indexes], teacher_features[positives_indexes].T),
                                                    p=alpha)
                        first_order = False
                    else:
                        loss_triplet += torch.norm(torch.matmul(student_features[queries_indexes], teacher_features[negatives_indexes].T) 
                                                   - torch.matmul(teacher_features[queries_indexes], teacher_features[negatives_indexes].T),
                                                    p=alpha)
                # CSD KL
                # for triplets in triplets_local_indexes:
                #     queries_indexes, positives_indexes, negatives_indexes = triplets.T
                #     loss_triplet += calculate_losses_KL(student_features[queries_indexes], teacher_features[queries_indexes], teacher_features, K=5, tau_q=1.0, tau_g=0.01)
        

            loss_triplet /= (args.train_batch_size * args.negs_num_per_query)
        
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()
            
            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet
        
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                      f"current batch triplet loss = {batch_loss:.4f}, " +
                      f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, student_model, teacher_model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    
    is_best = recalls[0] + recalls[1] > best_r5
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {
        "epoch_num": epoch_num, "student_model_state_dict": student_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_student_model.pth")
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[0] + recalls[1]:.1f}")
        best_r5 = recalls[0] + recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[0] + recalls[1]:.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break


logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_student_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["student_model_state_dict"]
student_model.load_state_dict(best_student_model_state_dict)

recalls, recalls_str = test.test(args, test_ds, student_model, teacher_model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")



