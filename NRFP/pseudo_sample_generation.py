import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import numpy as np


def getAuxData(aux_data, aux_label, num=5, N=128, random=True):
    data = []
    label = []

    length = int(len(aux_data) / N)
    if random:
        if length <= num:
            print('长度不够随机采样,执行不随机采样')
            start_ep = 0
        else:
            print('random sample')
            start_ep = np.random.randint(length - num)
    else:
        print('no random sample')
        start_ep = 0
    if length == 0:
        end_up = 1
    else:
        end_up = length
    print(f'length={length} start_ep={start_ep} end_up={end_up}')
    index = 0
    for ind in range(start_ep, end_up):
        if index > num:
            break
        temp_data = []
        temp_label = []
        temp_indices = []
        for i in range(N):
            idx = (ind * (N - 1)) + i
            if idx >= len(aux_data):  # 防止越界
                temp_data.append(aux_data[0].cpu().numpy())
                temp_label.append(aux_label[0].cpu().numpy())
                temp_indices.append(0)
            else:
                temp_data.append(aux_data[idx].cpu().numpy())
                temp_label.append(aux_label[idx].cpu().numpy())
                temp_indices.append(idx)
        try:
            temp_data = torch.tensor(np.asarray(temp_data))
            temp_label = torch.tensor(np.asarray(temp_label))
            if len(temp_data) == N:
                data.append(temp_data)
                label.append(temp_label)
            index += 1
        except Exception as e:
            print(f"Error occurred at index {index}: {e}")
            print('error1, skip')
            import traceback
            traceback.print_exc()

    return data, label


def cluster_features_bans(logits_bans, eps=2.0, min_samples=10):
    N, C = logits_bans.shape
    preds = np.argmax(logits_bans, axis=1)

    min_centers, max_centers = [], []
    min_clusters_samples, max_clusters_samples = [], []

    for c in range(C):
        idxs = np.where(preds == c)[0]
        if len(idxs) == 0:
            continue

        probs = logits_bans[idxs]
        max_probs = probs[:, c]

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(probs)
        labels = clustering.labels_

        best_min_cluster, min_score = None, float("inf")
        best_max_cluster, max_score = None, float("-inf")
        for label in set(labels):
            if label == -1:
                continue
            cluster_probs = max_probs[labels == label]
            avg_prob = cluster_probs.mean()
            if avg_prob < min_score:
                min_score = avg_prob
                best_min_cluster = label
            if avg_prob > max_score:
                max_score = avg_prob
                best_max_cluster = label

        if best_min_cluster is not None:
            cluster_points = probs[labels == best_min_cluster]
            cluster_idxs = idxs[labels == best_min_cluster]
            min_centers.append(cluster_points.mean(axis=0))
            min_clusters_samples.append(logits_bans[cluster_idxs[0]])

        if best_max_cluster is not None:
            cluster_points = probs[labels == best_max_cluster]
            cluster_idxs = idxs[labels == best_max_cluster]
            max_centers.append(cluster_points.mean(axis=0))
            max_clusters_samples.append(logits_bans[cluster_idxs[0]])

    return (
        np.array(min_centers),
        np.array(max_centers),
        np.array(min_clusters_samples),
        np.array(max_clusters_samples)
    )


def split_batch(data, label, n):
    N = len(data)
    batch_size = N // n  # batch 数量，丢弃余数
    data_splits, label_splits = [], []

    for i in range(batch_size):
        start = i * n
        end = (i + 1) * n
        data_splits.append(data[start:end])
        label_splits.append(label[start:end])

    return data_splits, label_splits


def FieldAlignment(data, label, optimizer, netF, netC, args, sample_num=5, sample_random=True):
    N,C,H,W = data.shape
    print(f'start domain alignment;')
    global data_three, label_three
    if N != 0:
        data_three, label_three = getAuxData(torch.cat([data], dim=0),
                                             torch.cat([label], dim=0), sample_num, args.class_num,
                                             random=sample_random)
        print(f'need compute:{len(data_three)}')

        for ind in range(len(data_three)):
            inputs = data_three[ind].cuda()
            labels = label_three[ind].cuda()

            adData = inputs.clone().detach()
            adData = adData.cuda()
            adData.requires_grad = True

            # 第二个模块-start
            netF.eval()
            netC.eval()

            iter_num = args.iter_num
            print(f'The {ind}-start domain shift compute; iter_num={iter_num}')
            for i in range(iter_num):
                _, features_test = netF(adData)
                output3 = netC(features_test)

                loss = -nn.CrossEntropyLoss()(output3, labels)

                adData.grad = None
                loss.backward(retain_graph=True)
                cgs = adData.grad

                cgsView = cgs.view(cgs.shape[0], -1)
                cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
                cgsView /= cgsnorms[:, np.newaxis]

                adData.data = adData.data - 0.01 * cgs
                adData.data = torch.clamp(adData.data, inputs.data - 0.3, inputs.data + 0.3)
            # 第二个模块-end

            netF.train()
            netC.train()

            # 第一个模块-start
            origin_data, origin_label = split_batch(data, label, args.class_num)
            if sample_num > 2:
                for i in range(1):
                    for n in range(len(origin_data)):
                        _, features_test = netF(origin_data[n])
                        origin_output = netC(features_test)

                        classLoss = nn.CrossEntropyLoss()(origin_output, origin_label[n])
                        optimizer.zero_grad()
                        classLoss.backward()
                        optimizer.step()
            # 第一个模块-end

            # 第二个模块-start
            for i in range(1):
                _, features_test = netF(adData)
                adv_outputs = netC(features_test)

                classLoss = nn.CrossEntropyLoss()(adv_outputs, labels)
                optimizer.zero_grad()
                classLoss.backward()
                optimizer.step()
            # 第二个模块-end

    print('domain alignment finish!')

