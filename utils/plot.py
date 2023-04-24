import matplotlib.pyplot as plt
import torchvision
from parser_my import args

def plot_roc(fpr, tpr, all_fpr, mean_tpr, auc, auc_macro, num_class):
    plt.figure()
    plt.plot(all_fpr, mean_tpr,
            label='macro-average ROC curve (area = {:0.4f})'
                ''.format(auc_macro),
            color='navy', linestyle=':', linewidth=4)

    for i in range(num_class):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'
                ''.format(i, auc[i].item()))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', {'size':20})
    plt.ylabel('True Positive Rate', {'size':20})
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(conf_matrix, num_class, classes):
    # 绘制混淆矩阵

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2	#数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(num_class):
        for y in range(num_class):
            info = int(conf_matrix[x, y])
            plt.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="white" if info > thresh else "black")
                    

    plt.yticks(range(num_class), classes)
    plt.xticks(range(num_class), classes)
    plt.xlabel('Prediction', {'size':20})
    plt.ylabel('Truth', {'size':20})
    plt.title('Confusion Matrix')
    plt.show()


def plot_tensor(tensor):
    for i in range(3):
        tensor[:,i,:,:] = tensor[:,i,:,:] * args.std[i] + args.mean[i]
    tensor = torchvision.utils.make_grid(tensor)
    np = tensor.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(20,20))
    plt.imshow(np)

