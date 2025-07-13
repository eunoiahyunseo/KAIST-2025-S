import lightning as L
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

class DigitDataModule(L.LightningDataModule):
    def __init__(self, dict_size: int, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # change pixel intensity to discrete values
                transforms.Lambda(lambda x: (x[0] * (dict_size - 2) + 1).long()),
                # we fill the padding with 1 since 0 is the mask token
                transforms.Pad((2, 2, 2, 2), fill=1, padding_mode="constant"),
            ]
        )

    def prepare_data(self):
        MNIST("MNIST", train=True, download=True)
        MNIST("MNIST", train=False, download=True)

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            full_set = MNIST(
                root="MNIST",
                train=True,
                transform=self.transform,
                download=True,
            )
            train_set_size = int(len(full_set) * 0.8)
            val_set_size = len(full_set) - train_set_size
            seed = torch.Generator().manual_seed(42)
            (
                self.train_set,
                self.val_set,
            ) = data.random_split(  # Split train/val datasets
                full_set, [train_set_size, val_set_size], generator=seed
            )
        elif stage == "test":
            self.test_set = MNIST(
                root="MNIST",
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=10
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=10
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=10
        )

if __name__ == "__main__":
    DICT_SIZE = 10
    MODEL_CHANNELS = 64
    

    dm = DigitDataModule(dict_size=DICT_SIZE)
    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
        break  # Just to check the first batch

    images, labels = next(iter(train_loader))

    print(f"이미지 배치 텐서 모양: {images.shape}")
    print(f"레이블 배치 텐서 모양: {labels.shape}")


    # 4. Matplotlib으로 이미지 시각화
    # --------------------------------
    # 4x4 그리드 생성
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    fig.suptitle("MNIST 데이터 샘플 시각화", fontsize=16)

    for i, ax in enumerate(axes.flat):
        # 16개 이미지만 표시
        if i < 16:
            # 텐서를 numpy 배열로 변환하고 채널 차원 제거
            img = np.squeeze(images[i].numpy())
            
            # 이미지 표시 (흑백 컬러맵)
            ax.imshow(img, cmap="gray")
            
            # 각 이미지 위에 레이블(정답 숫자) 표시
            ax.set_title(f"Label: {labels[i].item()}")
        
        # 축 정보 숨기기
        ax.axis("off")

    plt.tight_layout()
    plt.show()