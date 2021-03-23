import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from deep_ensemble.train import train



transforms_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])
transforms_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

train_dataset = CIFAR10(
    root='./data',
    train=True, 
	download=True,
    transform=transforms_train
)
test_dataset = CIFAR10(
    root='./data',
    train=False, 
	download=True,
    transform=transforms_test
)

def train_cifar10(
        model_save_dir,
        log_dir,
        model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_frac=1,
        num_epochs=300,
        train_batch_size=256,
        test_batch_size=256,
        decay_epoch=[80, 125],
        save_checkpoint=False,
        lr=0.1, 
	    momentum=0.9,
        weight_decay=1e-4,
        seed=10,
    ):
    train_len=int(train_frac * len(train_dataset))
    remain_len = len(train_dataset) - train_len
    torch.manual_seed(seed)
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset,
        [train_len, remain_len]
    )
    
    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_save_dir=model_save_dir,
        log_dir=log_dir,
        model=model,
        num_epochs=num_epochs,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        decay_epoch=decay_epoch,
        save_checkpoint=save_checkpoint,
        lr=lr, 
	    momentum=momentum,
        weight_decay=weight_decay,
    )