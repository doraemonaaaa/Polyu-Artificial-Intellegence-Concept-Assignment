import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter  # 导入 TensorBoard 的 SummaryWriter

data_path = '../data/MNIST'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
# print(torch.cuda.is_available())  # 查看是否有可用的 GPU
# print(torch.cuda.device_count())  # 查看可用的 GPU 数量
# print(torch.cuda.current_device())  # 查看当前设备 ID
# print(torch.cuda.get_device_name(0))  # 查看第一个 GPU 的名称

class ImageTransformerNet(nn.Module):
    """
    - 将 conv 输出视为 patch 序列（seq_len = 7*7），d_model = channels(32)
    - 正确实现多头自注意力（跨 patch 计算注意力）
    - 不增加显著参数量（只是重用线性投影）
    """
    def __init__(self, x_dim, y_dim, num_heads):
        super(ImageTransformerNet, self).__init__()
        self.type = 'ImageTransformer'
        self.num_heads = num_heads

        # 卷积提取局部特征，保留 channel 作为 d_model
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # patch 序列配置：7x7 -> seq_len=49, d_model=32
        self.seq_h = 7
        self.seq_w = 7
        self.seq_len = self.seq_h * self.seq_w
        self.d_model = 32

        if self.d_model % num_heads != 0:
            raise ValueError("num_heads must divide evenly into d_model")

        self.head_dim = self.d_model // num_heads

        # 每个头的线性投影（从 d_model -> head_dim）
        self.w_q_group = nn.ModuleList([nn.Linear(self.d_model, self.head_dim) for _ in range(num_heads)])
        self.w_k_group = nn.ModuleList([nn.Linear(self.d_model, self.head_dim) for _ in range(num_heads)])
        self.w_v_group = nn.ModuleList([nn.Linear(self.d_model, self.head_dim) for _ in range(num_heads)])

        # 输出线性层：把多头拼接回 d_model
        self.w_concat = nn.Linear(self.d_model, self.d_model)

        # 层归一化和分类头
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(0.2)
        self.w_out = nn.Linear(self.d_model, y_dim)

    def self_attention(self, q, k, v):
        # q,k,v: [batch, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, seq_len, seq_len]
        scaling = math.sqrt(k.size(-1))
        scores = scores / scaling
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # [batch, seq_len, head_dim]
        return out

    def position_encoding(self, batch_size, seq_len, d_model, device):
        # 标准的 sin-cos 位置编码，返回形状 [1, seq_len, d_model]
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, d_model]
        return pe

    def multi_head_attention(self, emb):
        # emb: [batch, seq_len, d_model]
        head_outs = []
        for h in range(self.num_heads):
            w_q = self.w_q_group[h]
            w_k = self.w_k_group[h]
            w_v = self.w_v_group[h]
            q = w_q(emb)  # [batch, seq_len, head_dim]
            k = w_k(emb)
            v = w_v(emb)
            out = self.self_attention(q, k, v)
            head_outs.append(out)
        # concat on last dim -> [batch, seq_len, d_model]
        all_heads = torch.cat(head_outs, dim=-1)
        # optional linear to mix heads (here we reuse a small linear)
        out = self.w_concat(all_heads)
        return out

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 32, 7, 7]

        b, c, h, w = x.size()
        # reshpae to patch sequence: [batch, seq_len, d_model]
        emb = x.view(b, c, h * w).permute(0, 2, 1).contiguous()  # [batch, seq_len, d_model]

        # add positional encoding
        pe = self.position_encoding(b, self.seq_len, self.d_model, x.device)  # [1, seq_len, d_model]
        emb = emb + pe

        # multi-head attention across patches
        attn_out = self.multi_head_attention(emb)  # [batch, seq_len, d_model]

        # Add & Norm
        emb_out = emb + attn_out
        emb_out = self.layer_norm(emb_out)
        emb_out = self.dropout(emb_out)

        # 池化到全局表示，然后分类
        pooled = emb_out.mean(dim=1)  # [batch, d_model]
        out = self.w_out(pooled)
        return out
        

def create_dataloader():
    # MNIST dataset

    train_dataset = torchvision.datasets.MNIST(root=data_path,
                                               train=True,
                                               download=True,
                                               transform=transforms.ToTensor())

    test_dataset = torchvision.datasets.MNIST(root=data_path,
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor())
    # Data loader
    # 增大 batch_size 有助于稳定训练（根据显存调整）
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True)

    print("MNIST 数据集下载完成！")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    return train_loader, test_loader

def train(train_loader, model, criterion, optimizer, num_epochs, scheduler=None):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        epoch_loss = 0  # 用来记录每个 epoch 的总损失
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), 1, 28, 28)  # [batch_size, 1, 28, 28]
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # 累加损失
                
            if (step + 1) % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * total_step + step)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))
        avg_loss = epoch_loss / total_step
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        # 学习率调度器在每个 epoch 结束后更新
        if scheduler is not None:
            scheduler.step()
            try:
                lr_now = scheduler.get_last_lr()[0]
            except Exception:
                lr_now = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR', lr_now, epoch)
        
        # 在每个epoch结束时记录准确率
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.size(0), 1, 28, 28)  # [batch_size, 1, 28, 28]
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.2f}%")
        writer.add_scalar('Accuracy/test', accuracy, epoch)


def test(test_loader, model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), 1, 28, 28)  # [batch_size, 1, 28, 28]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    lr = 1e-3
    head_num = 8
    epoch = 15
    # 创建 TensorBoard 写入器
    log_dir = f"runs/mnist_conv_transformer_Adam_lr{lr}_heads{head_num}_epoch{epoch}" 
    writer = SummaryWriter(log_dir=log_dir)
    ### step 1: prepare dataset and create dataloader
    train_loader, test_loader = create_dataloader()

    ### step 2: instantiate neural network and design model
    model = ImageTransformerNet(784, 10, num_heads=head_num).to(device)

    # Loss and optimizer (使用标签平滑和 AdamW + weight decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    ### step 3: train the model
    train(train_loader, model, criterion, optimizer, num_epochs=epoch, scheduler=scheduler)

    ### step 4: test the model
    test(test_loader, model)
    # 关闭 TensorBoard 写入器
    writer.close()
