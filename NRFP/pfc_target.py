import argparse
import os
import torch.optim as optim
import network
from utils import *
import os.path as osp
import random
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 工具函数定义
# ==========================================
def safe_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer

def set_batch_size(args):
    if args.office31:
        if args.dset in ['d2a', 'w2a', 'a2w']:
            args.batch_size = 84
        else:
            args.batch_size = 64
    elif args.home:
        if args.dset in ['a2r', 'r2p', 'p2c', 'p2r', 'c2r', 'a2p']:
            args.batch_size = 84
        else:
            args.batch_size = 64
    elif args.domainnet:
        args.batch_size = 256

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode="RGB"):
        imgs = make_dataset(image_list, labels)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def dset_target_load(args):
    train_bs = args.batch_size
    if args.office31:
        ss, tt = args.dset.split("2")[0], args.dset.split("2")[1]
        mapping = {"a": "amazon", "d": "dslr", "w": "webcam"}
        s, t = mapping[ss], mapping[tt]
        s_tr = open("./data/office/{}_list.txt".format(s)).readlines()
        t_tr = "./data/office/{}_list.txt".format(t)
        prep_dict = {"source": image_train(), "target": image_target(), "test": image_test()}
        train_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        test_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        train_target = ImageList_idx(open(t_tr).readlines(), transform=prep_dict["target"])
        test_target = ImageList_idx(open(t_tr).readlines(), transform=prep_dict["test"])
    elif args.home:
        ss, tt = args.dset.split("2")[0], args.dset.split("2")[1]
        mapping = {"a": "Art", "c": "Clipart", "p": "Product", "r": "Real_World"}
        s, t = mapping[ss], mapping[tt]
        s_tr = open("./data/office-home/{}.txt".format(s)).readlines()
        t_tr = "./data/office-home/{}.txt".format(t)
        prep_dict = {"source": image_train(), "target": image_target(), "test": image_test()}
        train_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        test_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        train_target = ImageList_idx(open(t_tr).readlines(), transform=prep_dict["target"])
        test_target = ImageList_idx(open(t_tr).readlines(), transform=prep_dict["test"])
    elif args.domainnet:
        ss, tt = args.dset.split("2")[0], args.dset.split("2")[1]
        mapping = {"s": "sketch", "c": "clipart", "p": "painting", "r": "real"}
        s, t = mapping[ss], mapping[tt]
        prefix = './data/domainnet-126/'
        s_tr = [prefix + line.strip() for line in open("./data/domainnet-126/{}.txt".format(s)).readlines()]
        t_tr = [prefix + line.strip() for line in open("./data/domainnet-126/{}.txt".format(t)).readlines()]
        prep_dict = {"source": image_train(), "target": image_target(), "test": image_test()}
        train_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        test_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        train_target = ImageList_idx(t_tr, transform=prep_dict["target"])
        test_target = ImageList_idx(t_tr, transform=prep_dict["test"])

    dset_loaders = {
        "source_tr": DataLoader(train_source, batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False),
        "source_te": DataLoader(test_source, batch_size=train_bs * 2, shuffle=True, num_workers=args.worker, drop_last=False),
        "target": DataLoader(train_target, batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False),
        "test": DataLoader(test_target, batch_size=train_bs * 3, shuffle=True, num_workers=args.worker, drop_last=False)
    }
    return dset_loaders

def target_adapt(args):
    # --- 耗时统计开始 ---
    exp_start_time = time.time()

    dset_loaders = dset_target_load(args)
    netF = network.ResNet_FE(class_num=args.class_num).cuda()
    oldC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # 路径拼接逻辑
    netF.load_state_dict(torch.load(osp.join(args.output_dir, "source_F.pt")))
    oldC.load_state_dict(torch.load(osp.join(args.output_dir, "source_C.pt")))

    # 伪标签预处理
    if getattr(args, 'pseudo_first', False):
        try:
            netF.eval(); oldC.eval()
            loader_tmp = dset_loaders['target']
            num_tmp = len(loader_tmp.dataset)
            confs = [0.0] * num_tmp
            iter_tmp = iter(loader_tmp)
            with torch.no_grad():
                for _ in range(len(loader_tmp)):
                    batch = safe_next(iter_tmp)
                    inputs, idxs = batch[0].cuda(), batch[-1]
                    _, feat = netF(inputs)
                    prob = nn.Softmax(dim=1)(oldC(feat))
                    topv, _ = torch.max(prob, dim=1)
                    for j, sample_idx in enumerate(idxs.cpu().tolist()):
                        confs[int(sample_idx)] = float(topv[j].item())
            
            thresh = float(getattr(args, 'pseudo_thresh', 0.9))
            trusted = [i for i, c in enumerate(confs) if c >= thresh]
            if len(trusted) > 0:
                ordered = trusted + [i for i in range(num_tmp) if i not in set(trusted)]
                class IndexSampler:
                    def __init__(self, indices): self.indices = indices
                    def __iter__(self): return iter(self.indices)
                    def __len__(self): return len(self.indices)
                dset_loaders['target'] = DataLoader(loader_tmp.dataset, batch_size=loader_tmp.batch_size, shuffle=False, sampler=IndexSampler(ordered), num_workers=loader_tmp.num_workers)
        except Exception as e: print(f"Pseudo Error: {e}")

    optimizer = op_copy(optim.SGD([
        {"params": netF.feature_layers.parameters(), "lr": args.lr * 0.1},
        {"params": netF.bottle.parameters(), "lr": args.lr},
        {"params": netF.bn.parameters(), "lr": args.lr},
        {"params": oldC.parameters(), "lr": args.lr * 0.1},
    ], momentum=0.9, weight_decay=5e-4, nesterov=True))

    loader = dset_loaders["target"]
    fea_bank = torch.randn(len(loader.dataset), args.bottleneck)
    score_bank = torch.randn(len(loader.dataset), args.class_num).cuda()

    netF.eval(); oldC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = safe_next(iter_test)
            inputs, indx = data[0].cuda(), data[-1]
            _, feat = netF(inputs)
            score_bank[indx] = nn.Softmax(-1)(oldC(feat)).detach()
            fea_bank[indx] = F.normalize(feat).cpu().detach()

    max_iter = args.max_epoch * len(loader) if getattr(args, 'max_iter', None) is None else int(args.max_iter)
    interval_iter = max(1, max_iter // args.interval)
    iter_num = 0

    while iter_num < max_iter:
        try: inputs_test, _, tar_idx = safe_next(iter_target)
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = safe_next(iter_target)

        if inputs_test.size(0) == 1: continue

        if iter_num % interval_iter == 0:
            netF.eval(); oldC.eval()
            mem_label = torch.from_numpy(obtain_label(dset_loaders['target'], netF, oldC)).cuda()
            netF.train(); oldC.train()

        iter_num += 1
        inputs_target = inputs_test.cuda()
        _, features_test = netF(inputs_target)
        output = oldC(features_test)
        softmax_out = nn.Softmax(dim=1)(output)
        
        fea_bank[tar_idx] = F.normalize(features_test).cpu().detach()
        score_bank[tar_idx] = softmax_out.detach()

        distance = F.normalize(features_test).cpu().detach() @ fea_bank.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
        score_near = score_bank[idx_near[:, 1:]].sum(dim=1)

        loss = torch.mean(F.kl_div(softmax_out, score_near, reduction="none").sum(1)) * args.sim_hyper
        loss += nn.CrossEntropyLoss()(output, mem_label[tar_idx]) * (1 - np.exp(-5 * (iter_num / max_iter)))
        
        # --- 修复 Device Mismatch 的核心修改点 ---
        # 显式指定 device=inputs_target.device 确保 torch.ones 在 GPU 上
        mask = (torch.ones((inputs_target.shape[0], inputs_target.shape[0]), device=inputs_target.device) - 
                (mem_label[tar_idx].view(-1, 1) == mem_label[tar_idx].view(1, -1)).float()).clamp(0, 1)
        loss += torch.mean((softmax_out @ softmax_out.T) * mask) * args.dis_hyper

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval(); oldC.eval()
            acc1, _ = cal_acc_(dset_loaders["test"], netF, oldC)
            log_str = f"Task: {args.dset}, Iter:{iter_num}/{max_iter}; Accuracy = {acc1*100:.2f}%"
            print(log_str)
            args.out_file.write(log_str + "\n"); args.out_file.flush()

    # --- 统计结束并保存 ---
    exp_end_time = time.time()
    total_sec = exp_end_time - exp_start_time
    duration = time.strftime("%H:%M:%S", time.gmtime(total_sec))
    summary = f"\n[Overhead] Total Time: {total_sec:.2f}s | Formatted: {duration}\n"
    print(summary)
    args.out_file.write(summary); args.out_file.flush()

def obtain_label(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        predict = []
        max_proto = F.adaptive_avg_pool2d(netF.proto[:, 0], (1, 1)).expand(-1, 2048, -1).squeeze()
        mean_proto = F.adaptive_avg_pool2d(netF.proto[:, 1], (1, 1)).expand(-1, 2048, -1).squeeze()
        for _ in range(len(loader)):
            data = safe_next(iter_test)
            if data is None: break
            inputs, labels = data[0].cuda(), data[1]
            feat, _ = netF(inputs)
            for i in range(feat.size(0)):
                feat_exp = F.adaptive_avg_pool2d(feat[i].unsqueeze(0), (1, 1)).repeat(max_proto.shape[0], 1, 1, 1).squeeze()
                fusion = netF.bn(netF.bottle(0.5 * max_proto * feat_exp + 0.5 * mean_proto * feat_exp))
                sim_scores = F.cosine_similarity(fusion, netC.fc.weight.data.cuda(), dim=1)
                _, p_idx = torch.max(sim_scores, dim=0)
                predict.append(p_idx.item())
    return torch.tensor(predict).numpy().astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PFC")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--worker", type=int, default=4)
    parser.add_argument("--dset", type=str, default="w2a")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--dis_hyper", type=float, default=0.6)
    parser.add_argument("--sim_hyper", type=float, default=1.6)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--layer", type=str, default="wn")
    parser.add_argument("--output", type=str, default="weight")
    parser.add_argument("--file", type=str, default="adaptation")
    parser.add_argument("--pseudo_first", action="store_true")
    parser.add_argument("--pseudo_thresh", type=float, default=0.9)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--office31", action="store_true")
    parser.add_argument("--domainnet", action="store_true")
    args = parser.parse_args()

    # 针对 Office31 的路径逻辑修正
    if args.office31 or args.dset in ['a2d', 'a2w', 'd2a', 'd2w', 'w2a', 'w2d']:
        args.office31, args.class_num, args.output = True, 31, "office31_weight"
    elif args.home: args.class_num, args.output = 65, "office_home_weight"
    elif args.domainnet: args.class_num, args.output = 126, "domainnet_weight"

    set_batch_size(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.output_dir = osp.join("./", args.output, "seed" + str(args.seed), args.dset)
    if not osp.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    args.out_file = open(osp.join(args.output_dir, args.file + ".txt"), "w")
    target_adapt(args)