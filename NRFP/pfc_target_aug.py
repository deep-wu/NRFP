import argparse
import os
import torch.optim as optim
import network
from utils import *
import os.path as osp
import random
import torchvision.transforms as T
from pseudo_sample_generation import FieldAlignment
import time

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
        args.batch_size = 256  # Set batch size to 256 for all DomainNet tasks

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer

class ImageList_idx(Dataset):
    def __init__(
        self, image_list, labels=None, transform=None, target_transform=None, mode="RGB"
    ):
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
        # 去掉路径前面的'.'
        path = path.replace("./", "")
        path = path.replace("data/domainnet-126", "")
        abs_path = "/24085404041/shot_Trans/" + path
        img = self.loader(abs_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index  # return index as well

    def __len__(self):
        return len(self.imgs)

def dset_target_load(args):
    train_bs = args.batch_size
    fix_folder = '/24085404041/shot_Trans'
    if args.office31 == True:  # and not args.home and not args.visda:
        ss = args.dset.split("2")[0]
        tt = args.dset.split("2")[1]
        if ss == "a":
            s = "amazon"
        elif ss == "d":
            s = "dslr"
        elif ss == "w":
            s = "webcam"

        if tt == "a":
            t = "amazon"
        elif tt == "d":
            t = "dslr"
        elif tt == "w":
            t = "webcam"

        s_tr, s_ts = f"{fix_folder}/data/office/{s}_list.txt", f"{fix_folder}/data/office/{s}_list.txt"

        txt_src = open(s_tr).readlines()

        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = f"{fix_folder}/data/office/{t}_list.txt", f"{fix_folder}/data/office/{t}_list.txt"
        prep_dict = {}
        prep_dict["source"] = image_train()
        prep_dict["target"] = image_target()
        prep_dict["test"] = image_test()
        train_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        test_source = ImageList_idx(s_ts, transform=prep_dict["source"])
        train_target = ImageList_idx(
            open(t_tr).readlines(), transform=prep_dict["target"]
        )
        test_target = ImageList_idx(open(t_ts).readlines(), transform=prep_dict["test"])

    # office home dataset
    elif args.home == True:
        ss = args.dset.split("2")[0]
        tt = args.dset.split("2")[1]
        if ss == "a":
            s = "Art"
        elif ss == "c":
            s = "Clipart"
        elif ss == "p":
            s = "Product"
        elif ss == "r":
            s = "Real_World"

        if tt == "a":
            t = "Art"
        elif tt == "c":
            t = "Clipart"
        elif tt == "p":
            t = "Product"
        elif tt == "r":
            t = "Real_World"

        s_tr, s_ts = f"{fix_folder}/data/office-home/{s}.txt", f"{fix_folder}/data/office-home/{s}.txt"

        txt_src = open(s_tr).readlines()

        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = f"{fix_folder}/data/office-home/{t}.txt", f"{fix_folder}/data/office-home/{t}.txt"
        prep_dict = {}
        prep_dict["source"] = image_train()
        prep_dict["target"] = image_target()
        prep_dict["test"] = image_test()
        train_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        test_source = ImageList_idx(s_ts, transform=prep_dict["source"])
        train_target = ImageList_idx(
            open(t_tr).readlines(), transform=prep_dict["target"]
        )
        test_target = ImageList_idx(open(t_ts).readlines(), transform=prep_dict["test"])

    elif args.domainnet == True:
        ss = args.dset.split("2")[0]
        tt = args.dset.split("2")[1]
        if ss == "s":
            s = "sketch"
        elif ss == "c":
            s = "clipart"
        elif ss == "p":
            s = "painting"
        elif ss == "r":
            s = "real"

        if tt == "s":
            t = "sketch"
        elif tt == "c":
            t = "clipart"
        elif tt == "p":
            t = "painting"
        elif tt == "r":
            t = "real"

        s_tr, s_ts = f"{fix_folder}/data/DomainNet/{s}_list.txt", f"{fix_folder}/data/DomainNet/{s}_list.txt"

        txt_src = open(s_tr).readlines()

        s_tr = txt_src
        s_ts = txt_src

        prefix = './data/domainnet-126/'
        s_tr = [prefix + line.strip() for line in s_tr]
        s_ts = [prefix + line.strip() for line in s_ts]

        t_tr, t_ts = f"{fix_folder}/data/DomainNet/{t}_list.txt", f"{fix_folder}/data/DomainNet/{t}_list.txt"

        target = open(t_tr).readlines()
        target_test = open(t_ts).readlines()
        target = [prefix + line.strip() for line in target]
        target_test = [prefix + line.strip() for line in target_test]

        prep_dict = {}
        prep_dict["source"] = image_train()
        prep_dict["target"] = image_target()
        prep_dict["test"] = image_test()
        train_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        test_source = ImageList_idx(s_ts, transform=prep_dict["source"])
        train_target = ImageList_idx(
            target, transform=prep_dict["target"]
        )
        test_target = ImageList_idx(target_test, transform=prep_dict["test"])

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(
        train_source,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["source_te"] = DataLoader(
        test_source,
        batch_size=train_bs * 2,  # 2
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )

    dset_loaders["target"] = DataLoader(
        train_target,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  # 3
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )

    g_aux = torch.Generator()
    g_aux.manual_seed(2025)
    aux_paths = open(args.aux_dataset_path).readlines()
    dset_loaders["aux2"] = DataLoader(ImageList_idx(aux_paths, transform=image_aux()),
                                      batch_size=train_bs,
                                      shuffle=True,
                                      num_workers=args.worker,
                                      drop_last=True,
                                      generator=g_aux)

    dset_loaders["aux"] = DataLoader(ImageList_idx(aux_paths, transform=image_aux()),
                                     batch_size=4,
                                     shuffle=True,
                                     num_workers=args.worker,
                                     drop_last=True,
                                     generator=g_aux)
    return dset_loaders

def image_aux(resize_size=256, crop_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def target_adapt(args):
    dset_loaders = dset_target_load(args)

    netF = network.ResNet_FE(class_num=args.class_num).cuda()
    oldC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    modelpath = args.output_dir + "/source_F.pt"
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + "/source_C.pt"
    oldC.load_state_dict(torch.load(modelpath))
    netF = netF.cuda()
    oldC = oldC.cuda()

    # 多卡
    netF = nn.DataParallel(netF).cuda()
    oldC = nn.DataParallel(oldC).cuda()

    optimizer = optim.SGD(
        [
            {"params": netF.module.feature_layers.parameters(), "lr": args.lr * 0.1},
            {"params": netF.module.bottle.parameters(), "lr": args.lr},
            {"params": netF.module.bn.parameters(), "lr": args.lr},
            {"params": oldC.module.parameters(), "lr": args.lr * 0.1},
        ],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    optimizer = op_copy(optimizer)

    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, args.bottleneck)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.train()
    oldC.train()

    for _ in range(5):
        for batchNo, (data, labels, _) in enumerate(dset_loaders["aux2"]):
            data = data.cuda()
            labels = labels.cuda()
            _, features_test = netF(data)
            output = oldC(features_test)
            classifier_loss = nn.CrossEntropyLoss()(output, labels)

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

    netF.eval()
    oldC.eval()

    with torch.no_grad():
        aux_data_list = []
        aux_label = []
        t = 0
        print(f'辅助数据集大小={len(dset_loaders["aux"])}')
        for batchNo, (data, labels, _) in enumerate(dset_loaders["aux"]):
            data = data.cuda()
            _, features_test = netF(data)
            output = oldC(features_test)
            pred = torch.argmax(output, dim=1)
            for i in range(len(labels)):
                if pred[i] == labels[i]:
                    t += 1
                aux_data_list.append(data[i].detach().cpu().clone().numpy())
                aux_label.append(labels[i].detach().cpu().clone().numpy())

        aux_data = torch.tensor(np.array(aux_data_list)).cuda()
        aux_label = torch.tensor(np.array(aux_label)).cuda()

        t = t / len(aux_data)
        print(f'精度为={t}')
        # office31上需要这个，其它数据集不需要
        # if t > 0.8:  # 说明源模型注意力集中
        #     weak_adversarial_enhancement = True
        # else:
        #     weak_adversarial_enhancement = False
        weak_adversarial_enhancement = False

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            _, output = netF(inputs)
            output_norm = F.normalize(output)
            outputs = oldC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    current_epoch = -1

    netF.train()
    oldC.train()


    while iter_num < max_iter:
        if iter_num % interval_iter == 0:
            current_epoch += 1
# 插件开关
        if iter_num % interval_iter == 0 and current_epoch > 0:
            start_time = time.time()
            if weak_adversarial_enhancement and current_epoch % 3 == 0:
                FieldAlignment(aux_data, aux_label,
                               optimizer, netF, oldC, args, 1,
                               False)
            if not weak_adversarial_enhancement:
                FieldAlignment(aux_data, aux_label,
                               optimizer, netF, oldC, args, 5,
                               True)
            elapsed_time = time.time() - start_time
            print(f"插件耗时 at epoch {current_epoch}, iter {iter_num}: {elapsed_time:.4f} seconds")
#插件开关

        netF.train()
        oldC.train()

        try:
            inputs_test, labels, tar_idx = next(iter_target)
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_target)

        if inputs_test.size(0) == 1:
            continue

        gamma_ce = 1 - np.exp(-5 * (iter_num / max_iter))

        inputs_test = inputs_test.cuda()

        if iter_num % interval_iter == 0:
            netF.eval()
            oldC.eval()
            mem_label = obtain_label(dset_loaders['target'], netF, oldC)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            oldC.train()

        iter_num += 1

        inputs_target = inputs_test.cuda()

        _, features_test = netF(inputs_target)
        output = oldC(features_test)
        softmax_out = nn.Softmax(dim=1)(output)

        output_f_norm = F.normalize(features_test)

        output_f_ = output_f_norm.cpu().detach().clone()

        pred_bs = softmax_out

        fea_bank[tar_idx] = output_f_.detach().clone().cpu()
        score_bank[tar_idx] = pred_bs.detach().clone()

        distance = output_f_ @ fea_bank.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)

        idx_near = idx_near[:, 1:]  # batch x K
        score_near = score_bank[idx_near]  # batch x K x C

        score_near = score_near.sum(dim=1)

        loss = torch.mean(
            F.kl_div(softmax_out, score_near, reduction="none").sum(1)
        ) * args.sim_hyper

        pred = mem_label[tar_idx]

        classifier_loss = nn.CrossEntropyLoss()(output, pred)
        loss += classifier_loss * gamma_ce

        mask = torch.ones((inputs_target.shape[0], inputs_target.shape[0]))
        mask_label = (pred.view(-1, 1) == pred.view(1,
                                                    -1)).float()  # Create a mask where elements with the same label are True

        mask = mask - mask_label.cpu()  # Subtract label mask
        mask = mask.clamp(0, 1)  # Ensure all values are between 0 and 1 (in case of any negative values)

        copy = softmax_out.T  # .detach().clone()  #

        dot_neg = softmax_out @ copy  # batch x batch
        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * args.dis_hyper

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            oldC.eval()

            acc1, _ = cal_acc_(dset_loaders["test"], netF, oldC)  # 1
            log_str = "Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%".format(
                args.dset, iter_num, max_iter, acc1 * 100
            )
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)


def obtain_label(loader, netF, netC):
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader)
        predict = []

        # Define predict list to store the predicted class indices
        max_proto = netF.module.proto[:, 0]
        max_proto = F.adaptive_avg_pool2d(max_proto, (1, 1)).expand(-1, 2048, -1).squeeze()

        mean_proto = netF.module.proto[:, 1]
        mean_proto = F.adaptive_avg_pool2d(mean_proto, (1, 1)).expand(-1, 2048, -1).squeeze()

        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feat, _ = netF(inputs)

            batchsize, _, _, _ = feat.size()

            mod_feas = []
            output = []

            for i in range(batchsize):

                feat_exp = feat[i].unsqueeze(0)
                feat_exp = F.adaptive_avg_pool2d(feat_exp, (1, 1)).repeat(max_proto.shape[0], 1, 1, 1).squeeze()

                fusion = 0.5*max_proto* feat_exp + 0.5*mean_proto* feat_exp

                fusion = netF.module.bn(netF.module.bottle(fusion))
                # Calculate the cosine similarity between the pooled fusion tensors and each class prototype in netC
                sim_scores = F.cosine_similarity(fusion, netC.module.fc.weight.data.cuda(), dim=1)

                # Find the class with the highest similarity score
                _, predicted_class = torch.max(sim_scores, dim=0)

                # Assign the fusion result of the most similar class as the final fusion result
                mod_feas.append(fusion[predicted_class])

                outputs = netC(fusion[predicted_class])
                output.append(outputs)

                predict.append(predicted_class.item())

            output = torch.stack(output, dim=0)
            mod_feas = torch.stack(mod_feas, dim=0)

            if start_test:
                all_fea = mod_feas.float().cpu()
                all_output = output.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, mod_feas.float().cpu()), 0)
                all_output = torch.cat((all_output, output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        predict = torch.tensor(predict)

    return predict.numpy().astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PFC"
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=50, help="maximum epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--worker", type=int, default=6, help="number of workers")
    parser.add_argument("--dset", type=str, default="a2d")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--seed", type=int, default=1993, help="random seed")
    parser.add_argument("--class_num", type=int, default=0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--dis_hyper", type=float, default=0.6)
    parser.add_argument("--sim_hyper", type=float, default=1.6)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="weight")
    parser.add_argument("--file", type=str, default="adaptation")
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--office31", action="store_true")
    parser.add_argument("--domainnet", action="store_true")
    parser.add_argument("--aux_dataset_path", type=str, default=None)
    parser.add_argument("--iter_num", type=int, default=30, help="iter_num")
    args = parser.parse_args()

    set_batch_size(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    if args.office31:
        args.class_num = 31
        args.output = "office31_weight"
        if args.aux_dataset_path == None:
            args.aux_dataset_path = '/24085404041/shot_Trans/data/office/aux_office_clean2.txt'
    elif args.home:
        args.class_num = 65
        args.output = "office_home_weight"
        if args.aux_dataset_path == None:
            args.aux_dataset_path = '/24085404041/shot_Trans/data/office-home/aux_list2.txt'  # 你需要有这个文件
    elif args.domainnet:
        args.class_num = 126
        args.output = "domainnet_weight"
        args.aux_dataset_path = '/24085404041/PFC/PFC-main/data/domainnet-126/class_images.txt'  # 你需要有这个文件

    print(f'辅助数据集路径={args.aux_dataset_path}')

    current_folder = "/24085404041/PFC/PFC-main/"
    args.output_dir = osp.join(
        current_folder, args.output, "seed" + str(args.seed), args.dset
    )
    if not osp.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    args.out_file = open(osp.join(args.output_dir, args.file + ".txt"), "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    target_adapt(args)
