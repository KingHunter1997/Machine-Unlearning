import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model_merger import ModelMerge
from resnet_graph import resnet18
from matching_functions import match_tensors_zipit, match_tensors_permute
from metric_calculators import CovarianceMetric, MeanMetric
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import gc
import time
from tqdm import tqdm

# 检查是否可以使用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 创建保存目录
save_dir = os.path.join(os.getcwd(), "Fine_tuned_save")
os.makedirs(save_dir, exist_ok=True)

# 下载预训练的 ResNet-18 模型
resnet18_original = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet18_original.fc.in_features  # 获取全连接层的输入特征数

# 加载 CIFAR-10 测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# 加载 CIFAR-10 训练集
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 获取数据集大小
dataset_size = len(dataset)
# 使用 1/10 数据作为验证集
validation_split = 0.1
shuffle_dataset = True

# 创建数据集索引并打乱
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(42)  # 设置随机种子以便复现
    np.random.shuffle(indices)

# 构建训练集和验证集的索引
train_indices, val_indices = indices[split:], indices[:split]

# 创建采样器
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# 构建 DataLoader
trainloader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
valloader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# 检查训练集和验证集的数据量
print(f"训练集大小: {len(train_indices)}")
print(f"验证集大小: {len(val_indices)}")


def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    # 使用 tqdm 包装 dataloader，显示进度条
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Testing Progress"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 读取之前训练好的参数，创建 20 个新模型
trained_models = []
for j in range(20):
    print(f'Processing model {j + 1}')
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(num_ftrs, 10)  # 修改输出层的大小为 10，适应 CIFAR-10
    model = model.to(device)

    # 加载保存的前 5% 权重参数
    saved_params = torch.load(os.path.join(save_dir, f'full_model_{j + 1}.pth'))
    model.load_state_dict(saved_params, strict=False)
    # accuracy = test_model(model, testloader, device)
    # print(f'Test accuracy for model {j + 1}: {accuracy:.2f}%')
    # 将模型转换为图对象
    graph = resnet18(model).graphify()
    trained_models.append(graph)
    print(len(trained_models))

def merge_multiple_models(graphs, dataloader, reduce_ratio=0.5, save_path=None, metrics_save_path=None):

    if len(graphs) < 2:
        raise ValueError("Need at least two models to merge.")

    # 使用所有传入的模型创建一个 ModelMerge 实例
    merger = ModelMerge(*graphs, device=device)

    print("Computing metrics...")
    merger.compute_metrics(dataloader, metric_classes={'covariance': CovarianceMetric, 'mean': MeanMetric})

    # 计算和应用合并操作
    merger.compute_transformations(match_tensors_zipit, reduce_ratio=reduce_ratio)
    merger.apply_transformations()

    # 获取合并后的模型状态字典
    merged_model_state_dict = merger.get_merged_state_dict()

    if merged_model_state_dict is None:
        raise ValueError("Merging failed. `merged_model` is None.")

    # 更新 merged_graph 为当前合并后的模型
    merged_graph = graphs[0]
    merged_graph.model.load_state_dict(merged_model_state_dict)

    # 保存合并后的模型
    if save_path:
        torch.save(merged_graph.model.state_dict(), save_path)
        print(f"Merged model saved to {save_path}")

    # 清空内存并暂停几秒
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    # 将合并后的模型包装为 Graph 对象并返回
    return resnet18(merged_graph.model).graphify()


# 一次性合并所有模型
# 合并所有模型
final_save_path = os.path.join(save_dir, 'final_merged_model.pth')
final_merged_model = merge_multiple_models(
    trained_models, dataloader=valloader, save_path=final_save_path)

# 测试合并后的模型
final_accuracy = test_model(final_merged_model.model, testloader, device)
print(f'Final merged model accuracy: {final_accuracy:.2f}%')