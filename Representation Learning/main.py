from dataloader import *
from params import *
from collections import Counter
import tensorflow as tf


def src():
    # ==1.Load dataset==
    data_set = DataLoader()
    train_data, test_data, = data_set.get_data()

    # ==2.initialize==
    A = []
    A_TEST = []
    if not IS_PATCH:
        # (SAMPLE_LENGTH, 1400)
        for key in train_data.keys():
            for item in train_data[key]:
                item = np.squeeze(item)
                A.append(item)
        A = np.array(A).T
    else:
        # (PATCH_NUM,PATCH_SAMPLE_LENGTH,1400)
        A = np.zeros(shape=(PATCH_NUM, PATCH_SAMPLE_LENGTH, 1400), dtype=np.float64)
        # test set
        A_TEST = np.zeros(shape=(PATCH_NUM, PATCH_SAMPLE_LENGTH, 1200), dtype=np.float64)  # 相同预处理的测试样本
        train_index = 0
        test_index = 0
        for key in train_data.keys():
            for item in train_data[key]:
                A[:, :, train_index] = item
                train_index += 1
        for key_test in test_data.keys():
            for item_test in test_data[key_test]:
                A_TEST[:, :, test_index] = item_test
                test_index += 1
        print("The shape of patch:", A.shape, A_TEST.shape)

    # == 3.Computing Sparse Expression of Test Data x with OMP Algorithm==
    Acc = {}                # 未分块每个人的准确率
    Acc_with_votes = {}     # 分块每个人的准确率
    # for each_person in test_data.keys():
    #     Acc[each_person] = []
    #     Acc_with_votes[each_person] = []
    if FOR_DICT:
        train_data = one_hot_for_dict(train_data)
    sample_index = 0    # 每一个测试样本的索引
    for each_person in test_data.keys():
        print("目前正在处理的类别：", each_person)
        print("waiting ...")
        pre_person = []  # 存储每个样本对应的预测类别
        predicted_person = []  # 分块情况下，该类所有样本被预测情况
        for each_sample in test_data[each_person]:
            # 对于每一个样本，分为分块处理与不分块处理的情况
            if IS_PATCH:
                pre_patch = []          # 每一个样本的预测情况
                for patch_index in range(PATCH_NUM):
                    patch_errors = {}
                    each_patch = A_TEST[patch_index, :, sample_index]
                    sample_index += 1
                    # 计算稀疏表达
                    x = omp(each_patch, A[patch_index], PATCH_SAMPLE_LENGTH)
                    index = 0  # 表示稀疏解中每个人对应的分块的起始坐标，每个分块14维
                    x = np.array(x)
                    # x = one_hot_likelihood(x)
                    # 应用字典将处理后的稀疏表达还原，并计算原后的向量和图像原始特征向量的距离
                    for potential in train_data.keys():
                        if FOR_DICT:
                            # 间接对初始字典处理
                            e = np.linalg.norm(each_patch[:, np.newaxis] - A[patch_index].dot(x * train_data[potential]))
                        else:
                            # 直接处理稀疏解，但效果较差
                            predicted_patch = A[patch_index, :, index:index + 14] * np.array(x)[index:index + 14]
                            index += 14
                            e = np.linalg.norm(each_patch[:, np.newaxis] - predicted_patch)
                        patch_errors[potential] = e  # 与某一类的误差
                    patch_predicted = min(patch_errors, key=patch_errors.get)
                    pre_patch.append(patch_predicted)
                selection = Counter(pre_patch)      # 计算该样本经过分块处理后，预测结果的分布情况
                predicted_person.append(selection.most_common(1)[0][0])     # 该样本的预测结果
            else:
                errors = []
                each_sample = np.squeeze(each_sample)  # (120,)
                x = omp(each_sample, A, SAMPLE_LENGTH)
                index = 0  # 表示稀疏解中每个人对应的分块的起始坐标，每个分块14维
                x = one_hot_likelihood(x)
                # 应用字典将处理后的稀疏表达还原，并计算原后的向量和图像原始特征向量的距离
                for potential in train_data.keys():
                    predicted_y = np.array(train_data[potential]).T * np.array(x)[index:index + 14]
                    index += 14
                    # L2 NORM
                    e = np.linalg.norm(each_sample[:, np.newaxis] - predicted_y)
                    errors[potential] = e  # 与某一类的误差
                # 残差最小的类为预测的类
                predicted = min(errors, key=errors.get)
                pre_person.append(predicted)
        # 保存准确率
        if IS_PATCH:
            Acc_with_votes[each_person] = accuracy(each_person, predicted_person)
        else:
            Acc[each_person] = accuracy(each_person, pre_person)

    # == 4. Output results ==
    draw_acc = []
    draw_class = []
    for k in test_data.keys():
        draw_acc.append(Acc_with_votes[k])
        draw_class.append(k)
    show_acc(draw_acc,draw_class)


src()
