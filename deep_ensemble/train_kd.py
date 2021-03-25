import torch
from torch import nn
import torch.nn.functional as F


def loss_fn_kd(
        outputs,
        labels,
        teacher_outputs,
        temperature,
        alpha,
    ):
    alpha = alpha
    T = temperature
    KD_loss = nn.KLDivLoss()(
        F.log_softmax(outputs/T, dim=1),
        F.softmax(teacher_outputs/T, dim=1),
    ) * (alpha * T * T) + F.cross_entropy(
            outputs,
            labels,
        ) * (1. - alpha)

    return KD_loss


def train_kd(
        train_dataset,
        test_dataset,
        model_save_dir,
        log_dir,
        model,
        teacher_model,
        num_epochs=300,
        train_batch_size=256,
        test_batch_size=256,
        decay_epoch=[80, 160],
        save_checkpoint=False,
        lr=0.1, 
	    momentum=0.9,
        weight_decay=1e-4,
        temperature=3,
        alpha=0.9
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
    model = model.to(device)
    teacher_model = teacher_model.to(device)
    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print('The number of parameters of model is', num_params)
    
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
    
    teacher_model.eval()

    global_steps = 0
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx_train, (inputs, targets) in tqdm(
                enumerate(train_loader),
                total=len(train_loader)
            ):
            global_steps += 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            loss = loss_fn_kd(
                outputs,
                targets,
                teacher_outputs,
                temperature,
                alpha
            )

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
        
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx_test, (inputs, targets) in tqdm(
                    enumerate(test_loader), total=len(test_loader)
                ):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn_kd(
                    outputs,
                    targets,
                    teacher_outputs,
                    temperature,
                    alpha
                )

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