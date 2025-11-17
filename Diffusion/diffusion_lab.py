import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 调试用：查看 CUDA 可用性及设备信息（需要时可取消注释）
# print(torch.cuda.is_available())  # 是否存在可用 GPU
# print(torch.cuda.device_count())  # 可用 GPU 的数量
# print(torch.cuda.current_device())  # 当前设备 ID
# print(torch.cuda.get_device_name(0))  # 第一个 GPU 的名称

class DiffusionNet(nn.Module):
    def __init__(self, beta_schedule):
        super(DiffusionNet, self).__init__()
        # 将 beta_schedule 注册为 buffer，使其随模型和设备一起移动
        self.register_buffer('beta_schedule', beta_schedule)
        # 计算 alpha 以及 alpha 的累乘（alpha_bar）
        alpha = 1.0 - beta_schedule
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', torch.cumprod(alpha, dim=0))
        # 编码器：更深的卷积 U-Net 风格主干网络
        # 卷积块：Conv -> GroupNorm -> SiLU
        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            )

        self.conv1 = conv_block(3, 64, stride=1)    # 32x32
        self.conv2 = conv_block(64, 128, stride=2)  # 16x16
        self.conv3 = conv_block(128, 256, stride=2) # 8x8
        self.conv4 = conv_block(256, 512, stride=2) # 4x4 (bottleneck)

        # 时间步嵌入的 MLP
        time_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )

        # 解码器：镜像编码器，使用转置卷积并配合跳跃连接
        # 将瓶颈特征上采样：512 -> 256（4x4 -> 8x8）
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 与编码器 e3 拼接后通道变为 512，然后继续上采样到 16x16
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 与编码器 e2 拼接后通道变为 256，然后继续上采样到 32x32
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 最后一个卷积：与编码器 e1 拼接后通道为 128，映射回 3 通道 RGB
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

        self.type = 'DiffusionNet'

    @staticmethod
    def get_timestep_embedding(timesteps, dim=128):
    # 来源于 Vaswani/Transformer 的正弦位置编码（做了适配）
    # timesteps: 形状为 (B,) 的长整型张量
        device = timesteps.device
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros(timesteps.size(0), 1, device=device)], dim=1)
        return emb

    def forward(self, x, t=None):
        """
        前向函数：给定输入 x（可以是带噪的 x_t）和时间步 t，预测噪声。
        如果 t 为 None，假定 t=0（干净图像）。
        返回预测的噪声，形状与输入 x 相同。
        """
        if t is None:
            t = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # 编码器部分
        e1 = self.conv1(x)   # (B,64,32,32)
        e2 = self.conv2(e1)  # (B,128,16,16)
        e3 = self.conv3(e2)  # (B,256,8,8)
        e4 = self.conv4(e3)  # (B,512,4,4) bottleneck

        # 时间步嵌入
        # time_dim 在 __init__ 中定义；这里假定维度匹配
        t_emb = self.get_timestep_embedding(t, dim=256)
        t_proj = self.time_mlp(t_emb)  # (B,512)
        t_proj = t_proj.view(-1, 512, 1, 1)

        # 将时间嵌入添加到瓶颈特征
        b = e4 + t_proj

        # 带跳跃连接的解码器
        d1 = torch.relu(self.deconv1(b))  # (B,256,8,8)
        d1_cat = torch.cat([d1, e3], dim=1)  # (B,256+256=512,8,8)
        d2 = torch.relu(self.deconv2(d1_cat))  # (B,128,16,16)
        d2_cat = torch.cat([d2, e2], dim=1)  # (B,128+128=256,16,16)
        d3 = torch.relu(self.deconv3(d2_cat))  # (B,64,32,32)
        d3_cat = torch.cat([d3, e1], dim=1)  # (B,64+64=128,32,32)
        out = self.final_conv(d3_cat)
        return out
    
    def forward_diffusion(self, x_0, t):
        """
        前向扩散过程：根据时间步 t 向 x_0 添加噪声，得到 x_t。
        """
        # 使用累计的 alpha_bar 来支持不同时间步的前向扩散（按样本选择对应的 alpha_bar）
        alpha_bar_t = self.alpha_bar[t].to(x_0.device)
        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.view(1, 1, 1, 1)
        elif alpha_bar_t.dim() == 1:
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def reverse_diffusion(self, x_t, t):
        """
        反向扩散过程：基于模型的噪声预测对图像进行去噪。
        x_t: 形状为 (B,C,H,W) 的张量
        t: 标量或形状为 (B,) 的时间步张量
        返回对 x_0 的预测（形状与 x_t 相同）。
        """
        # 使用 alpha_bar 和模型预测的噪声计算 x0 的估计
        alpha_bar_t = self.alpha_bar[t].to(x_t.device)
        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.view(1, 1, 1, 1)
        elif alpha_bar_t.dim() == 1:
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)

        pred_noise = self(x_t, t)
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        return x_0_pred

def create_dataloader():
    # 定义用于归一化数据的变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到大约 [-1,1]
    ])

    
    # 下载并加载训练集和测试集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 构造 DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    print("CIFAR10 数据集下载完成！")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    return train_loader, test_loader

def train(train_loader, model, criterion, optimizer, num_epochs, writer=None):
    # 训练模型并在每个 epoch 结束时保存检查点
    total_step = len(train_loader)
    global_step = 0
    for epoch in range(num_epochs):
        for step, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # 前向扩散：根据随机选择的时间步向图像添加噪声
            t = torch.randint(0, len(model.beta_schedule), (images.size(0),), device=device)
            x_t, noise = model.forward_diffusion(images, t)

            # 用模型预测噪声（训练目标为真实噪声）
            pred_noise = model(x_t, t)

            # 损失：预测噪声与真实噪声之间的loss
            loss = criterion(pred_noise, noise)

            # 用当前预测的噪声可计算 x0 的估计以便可视化
            x_0_pred = model.reverse_diffusion(x_t, t)
            # 反向传播并更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TensorBoard：记录训练损失和示例图像（偶尔）
            if writer is not None:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                if global_step % 500 == 0:
                    # log a small grid of originals and reconstructions (unnormalize to [0,1])
                    try:
                        # 构造原图和复原图的小网格用于可视化（反归一化到 [0,1]）
                        orig_grid = vutils.make_grid((images[:8] * 0.5 + 0.5).cpu(), nrow=4)
                        recon_grid = vutils.make_grid((x_0_pred[:8] * 0.5 + 0.5).cpu(), nrow=4)
                        writer.add_image('Images/Original', orig_grid, global_step)
                        writer.add_image('Images/Reconstruction', recon_grid, global_step)
                    except Exception:
                        pass
                global_step += 1

            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))
        # end of epoch: save checkpoint
        try:
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pt'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')
            if writer is not None:
                writer.add_text('checkpoint', ckpt_path, epoch + 1)
        except Exception as e:
            print(f'Warning: failed to save checkpoint: {e}')


def compute_psnr(a, b, data_range=1.0):
    # a, b: 范围为 [0,1] 的张量
    # 计算均方误差，并对 0 / NaN 做保护处理
    mse = torch.mean((a - b) ** 2)
    # If mse is NaN or zero, clamp to a small positive value to avoid NaN/inf
    if not torch.isfinite(mse):
        return float('nan')
    mse = mse.clamp_min(1e-12)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return float(psnr)


def test(test_loader, model, writer=None, num_samples=10):
    model.eval()  
    with torch.no_grad():
        for step, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # 通过迭代的反向扩散生成新图像（多步采样）
            T = len(model.beta_schedule)
            x = torch.randn_like(images).to(device)  # start from noise x_T
            # iterate from T-1 down to 0
            for t_idx in range(T - 1, -1, -1):
                t_vec = torch.full((images.size(0),), t_idx, device=device, dtype=torch.long)
                # 在当前时间步用模型预测噪声
                pred_noise = model(x, t_vec)

                beta_t = model.beta_schedule[t_idx].to(device)
                alpha_t = model.alpha[t_idx].to(device)
                alpha_bar_t = model.alpha_bar[t_idx].to(device)
                if t_idx > 0:
                    alpha_bar_prev = model.alpha_bar[t_idx - 1].to(device)
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)

                # 根据 DDPM 推导得到的后验均值 mu
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
                mu = coef1 * (x - coef2 * pred_noise)

                # 后验方差（标量） beta_tilde
                beta_tilde = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

                if t_idx > 0:
                    noise = torch.randn_like(x)
                    x = mu + torch.sqrt(beta_tilde) * noise
                else:
                    x = mu
            gen_batch = x

            # 先取用于预览的子集（生成图像与原图）
            n = min(num_samples, images.size(0))
            generated_images = gen_batch[:n]
            orig_images = images[:n]

            # 诊断信息：生成图像与原图的简要统计量（未反归一化前）
            try:
                gb = generated_images.detach()
                oi = orig_images.detach()
                print(f"GEN STAT min/max/mean/std: {gb.min().item():.4e}/{gb.max().item():.4e}/{gb.mean().item():.4e}/{gb.std().item():.4e}")
                print(f"ORIG STAT min/max/mean/std: {oi.min().item():.4e}/{oi.max().item():.4e}/{oi.mean().item():.4e}/{oi.std().item():.4e}")
                # print a few alpha_bar samples
                al = model.alpha_bar
                print(f"alpha_bar[0]={al[0].item():.4e}, alpha_bar[T//2]={al[len(al)//2].item():.4e}, alpha_bar[-1]={al[-1].item():.4e}")
            except Exception as e:
                print(f"Debug stats failed: {e}")

            # 保存生成网格以便检查
            try:
                os.makedirs('outputs', exist_ok=True)
                save_path = os.path.join('outputs', 'generated_preview.png')
                vutils.save_image((gen_batch[:n] * 0.5 + 0.5).clamp(0,1).cpu(), save_path, nrow=n)
                print(f"Saved generated preview to {save_path}")
            except Exception as e:
                print(f"Warning: failed to save generated preview: {e}")
            gen_batch = x

            # 取前 num_samples 张图像用于指标计算和可视化
            n = min(num_samples, images.size(0))
            generated_images = gen_batch[:n]
            orig_images = images[:n]

            # 将图像从 [-1,1] 反归一化到 [0,1]，以便计算指标与可视化
            gen_vis = (generated_images * 0.5 + 0.5)
            orig_vis = (orig_images * 0.5 + 0.5)

            # 数据清洗：替换 NaN/Inf，并裁剪到 [0,1]
            gen_vis = torch.where(torch.isfinite(gen_vis), gen_vis, torch.zeros_like(gen_vis))
            orig_vis = torch.where(torch.isfinite(orig_vis), orig_vis, torch.zeros_like(orig_vis))
            gen_vis = gen_vis.clamp(0, 1)
            orig_vis = orig_vis.clamp(0, 1)

            # 计算指标PNSR
            try:
                psnr_val = compute_psnr(gen_vis, orig_vis, data_range=1.0)
            except Exception:
                psnr_val = float('nan')

            print(f'Test PSNR (first batch): {psnr_val:.4f}')

            # 记录到 TensorBoard
            if writer is not None:
                writer.add_scalar('Metric/PSNR', psnr_val, step)
                try:
                    orig_grid = vutils.make_grid(orig_vis.cpu(), nrow=n)
                    gen_grid = vutils.make_grid(gen_vis.cpu(), nrow=n)
                    writer.add_image('Test/Original', orig_grid, 0)
                    writer.add_image('Test/Generated', gen_grid, 0)
                except Exception:
                    pass

            # 绘制生成图像与原图，便于交互式检查
            fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
            for i in range(n):
                img_o = orig_vis[i].cpu().permute(1, 2, 0).numpy()
                img_g = gen_vis[i].cpu().permute(1, 2, 0).numpy()
                # Ensure values are in [0,1] to avoid matplotlib clipping warnings
                img_o = np.clip(img_o, 0.0, 1.0)
                img_g = np.clip(img_g, 0.0, 1.0)

                axes[0, i].imshow(img_o)
                axes[0, i].axis('off')
                axes[0, i].set_title(f'Original {i+1}')

                axes[1, i].imshow(img_g)
                axes[1, i].axis('off')
                axes[1, i].set_title(f'Generated {i+1}')

            plt.show()
            break  # Only use the first batch for evaluation

if __name__ == '__main__':
    num_epochs = 10
    ### 步骤1：准备数据集并创建 DataLoader
    train_loader, test_loader = create_dataloader()

    ### 步骤2：实例化网络并构建模型
    beta_schedule = torch.linspace(1e-5, 0.1, 1000).to(device)
    model = DiffusionNet(beta_schedule).to(device)

    # 损失与优化器：对预测噪声使用 L2 loss
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 如果存在指定的检查点，则加载并直接运行推理（跳过训练）
    log_dir = os.path.join('runs', 'exp1')
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_path = r'C:\Users\彭宇航\Downloads\Lab\checkpoints\big\checkpoint_epoch_100.pt'
    if os.path.exists(ckpt_path):
        print(f'Found checkpoint at {ckpt_path}, loading for inference...')
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print('Loaded model state dict from checkpoint.')
        except Exception as e:
            print(f'Warning: failed to load state_dict directly: {e}. Trying full load...')
            model = torch.load(ckpt_path, map_location=device)

        # 运行测试/推理
        test(test_loader, model, writer=writer)
    else:
        ### 步骤3：训练模型
        train(train_loader, model, criterion, optimizer, num_epochs=num_epochs, writer=writer)

        ### 步骤4：测试模型
        test(test_loader, model, writer=writer)
