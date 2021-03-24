import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm



def train(
        train_dataset,
        test_dataset,
        model_save_dir,
        log_dir,
        model,
        num_epochs=300,
        train_batch_size=256,
        test_batch_size=256,
        decay_epoch=[80, 125],
        save_checkpoint=False,
        lr=0.1, 
	    momentum=0.9,
        weight_decay=1e-4,
    ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(model_save_dir):
        print('making model folder')
        os.mkdir(model_save_dir)
    if not os.path.exists(log_dir):
        print('making log folder')
        os.mkdir(log_dir)

    train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size, 
	shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size, 
        shuffle=False
    )


    net = model.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr, 
	    momentum=momentum,
        weight_decay=weight_decay
    )
    step_lr_scheduler = lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[epoch * train_batch_size for epoch in decay_epoch],
    gamma=0.1
    )

    global_steps = 0
    best_acc = 0
    for epoch in range(num_epochs):
        net.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx_train, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            global_steps += 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_lr_scheduler.step()


            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100 * correct / total
        print(
            f'''
            train epoch : {epoch} |
            loss: {train_loss/(batch_idx_train+1):.3f} |
            train_acc: {train_acc:.3f}'''
        )

        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx_test, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100 * correct / total
        print(
            f'''
            test epoch : {epoch} |
            loss: {train_loss/(batch_idx_test+1):.3f} |
            test_acc: {test_acc:.3f}'''
        )

        with open(f'{log_dir}/log.txt', 'a') as f:
            f.write(
                f'{epoch}\t{train_acc}\t{train_loss/(batch_idx_train+1)}\t{test_acc} \t {test_loss/(batch_idx_test+1)}\n'
            )

        if save_checkpoint:
            model_path=f'{model_save_dir}/ckpt_epoch_{epoch}.pth'
            torch.save(net.state_dict(), model_path)

        if test_acc > best_acc:
            best_acc = test_acc
            model_path=f'{model_save_dir}/best_model.pth'
            torch.save(net.state_dict(), model_path)

        print('best test accuracy is ', best_acc)