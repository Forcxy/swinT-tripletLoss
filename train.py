import torch
from swin_triple.utils.config import config
from models.swin_transformer import SwinFeatureExtractor
from models.triplet_loss import TripletLoss
from utils.dataset import get_dataloaders
from utils.trainer import TripletTrainer
from swin_triple.models.model  import swin_tiny_patch4_window7_224 as create_model
import os
import torch.optim as optim



def load_swin_weights(model, weights_path: str, freeze_layers=True):
    # 1. 验证权重文件存在
    assert os.path.exists(weights_path), f"weights file: '{weights_path}' not exist."

    # 2. 加载权重（保持原有格式处理）
    weights_dict = torch.load(weights_path, map_location='cpu')["model"]

    # 3. 移除分类头权重（完全复制原有逻辑）
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]

    # 4. 加载权重到backbone（非严格模式）
    load_result = model.load_state_dict(weights_dict, strict=False)
    print(f"权重加载结果: {load_result}")

    # 5. 冻结/解冻层（完全复制原有逻辑）
    if freeze_layers:
        print("\n冻结层配置:")
        for name, param in model.named_parameters():
            if not any(x in name for x in ["layers.2", "layers.3", "projection"]):  # 修改为projection
                param.requires_grad_(False)
            else:
                print("training: ", name)

def main():
    # 准备数据
    train_loader, val_loader = get_dataloaders(
        config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # 初始化模型
    # model = SwinFeatureExtractor(
    #     model_name=config.MODEL_NAME,
    #     pretrained=config.PRETRAINED,
    #     embedding_size=config.EMBEDDING_SIZE
    # )

    model = create_model(embedding_size=config.EMBEDDING_SIZE).to(config.DEVICE)
    # 加载预训练权重
    load_swin_weights(model, config.PRETRAINED_WEIGHTS, config.USE_PRE_WEIGHTS)

    # 初始化损失函数和优化器
    criterion = TripletLoss(
        margin=config.MARGIN,
        distance=config.DISTANCE
    )

    # 优化器参数过滤
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=config.LEARNING_RATE,
    #     weight_decay=config.WEIGHT_DECAY
    # )

    # 初始化训练器
    trainer = TripletTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=config.DEVICE,
        save_dir=config.SAVE_DIR
    )

    # 训练循环
    best_loss = float("inf")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.evaluate(val_loader)

        # 打印日志
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        # 保存日志和模型
        trainer._save_log(epoch, train_loss, train_acc, val_loss, val_acc)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            trainer.save_checkpoint(epoch, is_best)
        else:
            trainer.save_checkpoint(epoch)


if __name__ == "__main__":
    main()