import gzip
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

# 检查是否可以使用 GPU，如果有 GPU 可用则使用，否则使用 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 加载 CIFAR-10 训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)
# 将数据集的索引乱序，以便后续数据划分
indices = torch.randperm(len(trainset)).tolist()

# 计算每个子集的大小，并基于乱序后的索引将数据集划分为 20 个子集
split_size = len(trainset) // 20
subsets = [Subset(trainset, indices[i * split_size:(i + 1) * split_size]) for i in range(20)]

# 下载预训练的 ResNet-18 模型
resnet18_original = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet18_original.fc.in_features
resnet18_original.fc = nn.Linear(num_ftrs, 10)
resnet18_original.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 创建保存目录
save_dir = os.path.join(os.getcwd(), "Fine_tuned_save")
os.makedirs(save_dir, exist_ok=True)

# 定义测试函数
def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    # 使用 tqdm 显示进度条
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

# 提取前 5% 权重参数的函数
def get_top_5_percent_params(model):
    all_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    all_weights = torch.cat([param.view(-1) for _, param in all_params])
    total_weight_count = all_weights.numel()
    top_5_percent_count = int(total_weight_count * 0.05)
    _, top_5_percent_indices = torch.topk(all_weights.abs(), top_5_percent_count)

    sorted_indices_set = set(top_5_percent_indices.tolist())
    params_to_train = []
    current_index = 0

    for name, param in all_params:
        param_size = param.numel()
        param_indices = set(range(current_index, current_index + param_size))

        if len(param_indices.intersection(sorted_indices_set)) > 0:
            params_to_train.append((name, param))

        current_index += param_size

    return params_to_train

# 初始化模型函数
def initialize_model():
    resnet18_original = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = resnet18_original.fc.in_features
    resnet18_original.fc = nn.Linear(num_ftrs, 10)
    resnet18_original.to(device)
    return resnet18_original

# 冻结模型的其他参数
def freeze_parameters_except_top_5(model, top_5_percent_params):
    top_5_percent_param_names = {name for name, _ in top_5_percent_params}
    for name, param in model.named_parameters():
        if name not in top_5_percent_param_names:
            param.requires_grad = False

# 保存前 5% 权重参数到文件，仅保存更新过的参数名和值，并压缩文件
def save_top_5_percent_params_minimal(model, params_to_save, filepath):
    params_dict = {name: param.data.cpu().clone() for name, param in params_to_save}
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(params_dict, f)

# 保存完整模型参数到文件，确保加载时不丢失任何信息
def save_full_model_state(model, filepath):
    torch.save(model.state_dict(), filepath)

# 平行训练并保存每个模型的前 5% 参数
resnet18_updated = initialize_model()

# 遍历每个子集进行训练和测试
for j, subset in enumerate(subsets):
    trainloader = DataLoader(subset, batch_size=32, shuffle=True)
    testloader_subset = DataLoader(subset, batch_size=32, shuffle=False)

    resnet18_updated.load_state_dict(resnet18_original.state_dict())
    top_5_percent_params = get_top_5_percent_params(resnet18_updated)
    freeze_parameters_except_top_5(resnet18_updated, top_5_percent_params)

    # 保存训练前的参数
    params_before = {name: param.clone() for name, param in resnet18_updated.named_parameters()}

    optimizer = optim.SGD([param for name, param in top_5_percent_params], lr=0.005, momentum=0.9)

    for epoch in range(7):
        resnet18_updated.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f'Training Batch {j + 1} Epoch {epoch + 1}')):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = resnet18_updated(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Training Batch {j + 1}, Epoch {epoch + 1}, Average loss: {running_loss / len(trainloader):.4f}')

    # 检查训练后的参数是否有变化
    for name, param in resnet18_updated.named_parameters():
        if not param.requires_grad:  # 只检查被冻结的参数
            print(name, param)
            if not torch.equal(params_before[name], param):
                print(f"Warning: Parameter {name} has changed but should be frozen.")

    # 保存仅更新过的前 5% 参数名和值并压缩
    save_top_5_percent_params_minimal(resnet18_updated, top_5_percent_params,
                                      os.path.join(save_dir, f'save_batch{j + 1}_top_5_percent_version_3.pkl.gz'))

    # 保存完整模型参数以确保完整性
    save_full_model_state(resnet18_updated, os.path.join(save_dir, f'full_model_{j + 1}.pth'))

    # 训练完成后测试该子集模型
    accuracy = test_model(resnet18_updated, testloader, device)
    print(f'Test accuracy for model {j + 1}: {accuracy:.2f}%')

    # 比较文件大小
    full_model_path = os.path.join(save_dir, f'full_model_{j + 1}.pth')
    sparse_model_path = os.path.join(save_dir, f'save_batch{j + 1}_top_5_percent_version_3.pkl.gz')

    full_size = os.path.getsize(full_model_path) / (1024 * 1024)  # 转换为 MB
    sparse_size = os.path.getsize(sparse_model_path) / (1024 * 1024)  # 转换为 MB

    print(f"Full model size: {full_size:.2f} MB")
    print(f"Sparse model size: {sparse_size:.2f} MB")

print('Finished Training and Testing')

# # 完整训练所有参数的模型，进行对比实验
# resnet18_full_training = initialize_model()  # 初始化完整模型
#
# # 遍历每个子集进行完整训练和测试
# for j, subset in enumerate(subsets):
#     trainloader = DataLoader(subset, batch_size=32, shuffle=True)
#     testloader_subset = DataLoader(subset, batch_size=32, shuffle=False)
#
#     # 重新加载原始的权重参数
#     resnet18_full_training.load_state_dict(resnet18_original.state_dict())
#
#     # 定义优化器，包含所有参数
#     optimizer = optim.SGD(resnet18_full_training.parameters(), lr=0.001, momentum=0.9)
#
#     # 完整训练每个子集的模型
#     for epoch in range(5):
#         resnet18_full_training.train()
#         running_loss = 0.0
#         for i, data in enumerate(tqdm(trainloader, desc=f'Full Training Batch {j + 1} Epoch {epoch + 1}')):
#             inputs, labels = data[0].to(device), data[1].to(device)
#             optimizer.zero_grad()
#             outputs = resnet18_full_training(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         print(f'Full Training Batch {j + 1}, Epoch {epoch + 1}, Average loss: {running_loss / len(trainloader):.4f}')
#
#     # 测试完整训练的模型
#     full_accuracy = test_model(resnet18_full_training, testloader, device)
#     print(f'Test accuracy for fully trained model {j + 1}: {full_accuracy:.2f}%')
#
#     # 比较文件大小（与之前冻结 95% 的模型相比）
#     full_model_path = os.path.join(save_dir, f'full_trained_model_{j + 1}.pth')
#     torch.save(resnet18_full_training.state_dict(), full_model_path)
#
#     full_size = os.path.getsize(full_model_path) / (1024 * 1024)  # 转换为 MB
#     print(f"Full trained model size: {full_size:.2f} MB")
#
# print('Finished Full Training and Testing')
