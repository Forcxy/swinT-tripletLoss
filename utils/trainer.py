import torch
import torch.optim as optim
from tqdm import tqdm
import os
import json
from datetime import datetime


class TripletTrainer:
    def __init__(self, model, criterion, optimizer, device, save_dir):
        """
        初始化三元组训练器

        Args:
            model (nn.Module): 要训练的神经网络模型（需支持特征提取）
            criterion (nn.Module): 损失函数（如TripletMarginLoss）
            optimizer (torch.optim): 优化器（如Adam/SGD）
            device (torch.device): 计算设备（'cuda'/'cpu'）
            save_dir (str): 模型和日志的保存路径
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 初始化日志文件
        self.log_file = os.path.join(save_dir, "training_log.json")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump({"training_log": []}, f)

    def _calculate_accuracy(self, anchor_feat, positive_feat, negative_feat):
        """计算正样本距离小于负样本距离的比例（范围0.0~1.0）、不需要加margin"""
        d_pos = torch.norm(anchor_feat - positive_feat, p=2, dim=1)  # L2距离
        d_neg = torch.norm(anchor_feat - negative_feat, p=2, dim=1)
        correct = (d_pos < d_neg).float().mean().item()  # 直接计算比例
        return correct

    def train_epoch(self, train_loader, epoch):
        """训练单个epoch，返回平均loss和acc"""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for anchor, positive, negative in pbar:
            # 1. 数据迁移到设备
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            # 2. 前向传播
            anchor_feat = self.model(anchor)
            positive_feat = self.model(positive)
            negative_feat = self.model(negative)

            # 3. 计算损失和准确率
            loss = self.criterion(anchor_feat, positive_feat, negative_feat)
            acc = self._calculate_accuracy(anchor_feat, positive_feat, negative_feat)

            # 4. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 5. 累计统计
            total_loss += loss.item()
            total_acc += acc

            # 6. 实时显示当前batch指标
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc * 100:.2f}%'
            })

        # 7. 计算epoch平均指标
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        # 打印epoch结果
        print(f'Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc * 100:.2f}%')
        return avg_loss, avg_acc

    def evaluate(self, val_loader):
        """验证集评估，返回平均loss和acc"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0

        with torch.no_grad():
            for anchor, positive, negative in tqdm(val_loader, desc='Validating', leave=False):
                # 1. 数据迁移到设备
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # 2. 前向传播
                anchor_feat = self.model(anchor)
                positive_feat = self.model(positive)
                negative_feat = self.model(negative)

                # 3. 计算指标
                total_loss += self.criterion(anchor_feat, positive_feat, negative_feat).item()
                total_acc += self._calculate_accuracy(anchor_feat, positive_feat, negative_feat)

        # 4. 计算平均指标
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_acc / len(val_loader)

        # 打印验证结果
        print(f'Validation | Loss: {avg_loss:.4f} | Acc: {avg_acc * 100:.2f}%')
        return avg_loss, avg_acc

    def _save_log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """保存日志到JSON文件"""
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        with open(self.log_file, 'r+') as f:
            data = json.load(f)
            data["training_log"].append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=4)

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

        filename = os.path.join(self.save_dir, f"checkpoint_{epoch}.pth")
        torch.save(state, filename)

        if is_best:
            best_filename = os.path.join(self.save_dir, "model_best.pth")
            torch.save(state, best_filename)