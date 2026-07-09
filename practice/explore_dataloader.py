"""
目标 1 实践：数据流水线探索
对应章节：1.1 ~ 1.5

运行方式（在项目根目录下）：
    python practice/explore_dataloader.py

第一次运行会自动下载 CIFAR-100 到 ./data/
"""

import torch

from conf import settings
from utils import get_training_dataloader, get_test_dataloader


def section_a_dataset():
    """Section A — 1.1 + 1.2：认识 Dataset，读取单条样本"""
    print("=" * 50)
    print("[Section A] Dataset 单条样本")
    print("=" * 50)

    # TODO A-1: 调用 get_training_dataloader，创建训练集 DataLoader
    #   提示：传入 settings 中的 mean / std，batch_size 和 num_workers 可先设 128 和 0
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, batch_size=128, num_workers=0) #shuffle = true 

    # TODO A-2: 从 DataLoader 取出其内部的 Dataset 对象
    #   提示：DataLoader 有一个 .dataset 属性
    dataset = train_loader.dataset  # ← 替换这行

    # TODO A-3: 打印训练集总样本数（期望 50000）
    print(f"训练集样本数: {len(dataset)}")

    # TODO A-4: 取第 0 条样本，解包为 image 和 label
    #   提示：dataset[0] 返回一个元组
    image, label = dataset[0]  # 项目中的dataset返回的是元组(image, label)

    # TODO A-5: 打印 label 的值、类型，以及 image 的类型和 shape
    print(f"label 值: {label}, 类型: {type(label)}")
    print(f"image 类型: {image.dtype}, shape: {image.shape}")

    print()


def section_b_transforms():
    """Section B — 1.4：观察 transforms 对单条样本的影响"""
    print("=" * 50)
    print("[Section B] transforms 对比")
    print("=" * 50)

    # TODO B-1: 再次创建 train_loader（或与 Section A 共用，你来决定）
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, batch_size=128, num_workers=0, shuffle = False
    )

    # TODO B-2: 取一个 batch，从中拿出第一张图 image_0
    #   提示：next(iter(train_loader)) 可以取出一个 batch
    images, labels = next(iter(train_loader))  # ← 替换这行
    image_0 = images[0]  # ← 替换这行

    # TODO B-3: 打印 image_0 的 shape、dtype、最小值、最大值
    print(f"train transform 后 shape: {image_0.shape}")
    print(f"dtype: {image_0.dtype}, min: {image_0.min().item()}, max: {image_0.max().item()}")
    print(f"labels: {labels}")

    # TODO B-4（可选）: 创建 test_loader，取一个 batch 的第一张图，同样打印 shape / min / max
    #   思考：test 的 min/max 和 train 有什么不同？为什么？
    print("--- test ---")
    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, batch_size=128, num_workers=0, shuffle=False
    )
    images_t, labels_t = next(iter(test_loader))
    images_t_0 = images_t[0]
    print(f"test未经数据增强transform后的shape: {images_t_0.shape}")
    print(f"dtype: {images_t_0.dtype}, min: {images_t_0.min().item()}, max: {images_t_0.max().item()}")
    print(f"labels_t: {labels_t}")

    print()


def section_c_dataloader_batch():
    """Section C — 1.3 + 1.5：DataLoader 打包 batch，确认 NCHW"""
    print("=" * 50)
    print("[Section C] DataLoader batch 与 NCHW")
    print("=" * 50)

    # TODO C-1: 创建 train_loader（batch_size=128, num_workers=0）
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, batch_size=128, num_workers=0, shuffle=True
    )  

    it = iter(train_loader)
    print("-" * 10, "First run", '-' * 10)
    # TODO C-2: 取出一个 batch，解包为 images 和 labels
    images1, labels1 = next(it)  

    # TODO C-3: 打印 images 和 labels 的 shape
    #   思考：四个维度 (128, 3, 32, 32) 分别对应 NCHW 的什么？
    print(f"images shape: {images1.shape}")
    print(f"labels shape: {labels1.shape}")

    # TODO C-4: 打印 labels 的前 5 个值，确认是 0~99 的整数
    print(f"labels 前 5 个: {labels1[:5]}")

    print()
    print("-" * 10, "Second run", '-' * 10)
    images2, labels2 = next(it)  

    print(f"images shape: {images2.shape}")
    print(f"labels shape: {labels2.shape}")

    print(f"labels 前 5 个: {labels2[:5]}")


def section_d_normalize_range():
    """Section D — 1.4 + 1.5：Normalize 后的数值范围"""
    print("=" * 50)
    print("[Section D] Normalize 后的数值范围")
    print("=" * 50)

    # TODO D-1: 创建 train_loader，取一个 batch 的 images
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, batch_size=128, num_workers=0
    )  # ← 替换这行
    images, _ = next(iter(train_loader))  # ← 替换这行

    # TODO D-2: 打印整个 batch 的 min 和 max
    #   提示：tensor 有 .min() 和 .max() 方法
    #   思考：Normalize 后数值还在 0~1 吗？如果不是，正常吗？
    print(f"batch min: {images.min().item()}, max: {images.max().item()}")

    print()


def section_e_random_augment():
    """Section E — 1.3 + 1.4（可选）：验证训练集随机增强"""
    print("=" * 50)
    print("[Section E] 随机增强验证（可选）")
    print("=" * 50)

    # TODO E-1: 创建 train_loader
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, batch_size=128, num_workers=0
    )  # ← 替换这行
    dataset = train_loader.dataset  # ← 替换这行

    # TODO E-2: 对 dataset[0] 连续取两次，得到 image_a 和 image_b
    #   提示：每次访问 dataset[0] 都会重新执行 transform
    image_a, _ = dataset[0]  # ← 替换这行
    image_b, _ = dataset[0]  # ← 替换这行

    # TODO E-3: 判断两次结果是否完全相同
    #   提示：torch.equal(a, b) 可以比较两个 tensor 是否完全一致
    is_same = torch.equal(image_a, image_b)  # ← 替换这行
    print(f"同一张图两次 transform 结果相同: {is_same}")
    print(f"（训练集期望为 False，说明随机增强生效）")

    print(f"BONUS: 查看test集是否有随机增强")
    test_loader=get_test_dataloader(
        settings.CIFAR100_TEST_MEAN, settings.CIFAR100_TEST_STD, batch_size=128, num_workers=0, shuffle=False
    )
    dataset_t = test_loader.dataset
    image_t_a, _ = dataset_t[0]
    image_t_b, _ = dataset_t[0]
    print(f"Is test the same? : {torch.equal(image_t_a, image_t_b)}")

    print()


def main():
    # section_a_dataset()
    # section_b_transforms()
    # section_c_dataloader_batch()
    # section_d_normalize_range()
    section_e_random_augment()
    # print("全部 Section 完成。")


if __name__ == "__main__":
    main()
