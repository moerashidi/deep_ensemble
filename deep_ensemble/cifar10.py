import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from deep_ensemble.train_resnet import train_resnet



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

def train_cifar10_resnet(
        model_save_dir,
        log_dir,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_epochs=300,
        train_batch_size=256,
        test_batch_size=256,
        decay_epoch=[80, 125],
        save_checkpoint=False,
        lr=0.1, 
	    momentum=0.9,
        weight_decay=1e-4,
    ):
    train_resnet(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_save_dir=model_save_dir,
        log_dir=log_dir,
        num_epochs=num_epochs,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        decay_epoch=decay_epoch,
        save_checkpoint=save_checkpoint,
        lr=lr, 
	    momentum=momentum,
        weight_decay=weight_decay,
    )