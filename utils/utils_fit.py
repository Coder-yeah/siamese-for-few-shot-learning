import os

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval,
                  Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_accuracy = 0

    val_loss = 0
    val_total_accuracy = 0
    # 是否单机多卡分布式运行， 默认rank=0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]            # 数据由（（图1，图2），标签）组成
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(images)               # 送图片进去计算
            output = loss(outputs, targets)             # 计算损失

            output.backward()                           # 反向传播
            optimizer.step()                            # 优化器
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                output = loss(outputs, targets)
            # ----------------------#
            #   多卡并行的反向传播
            # ----------------------#
            scaler.scale(output).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            # 把outputs映射到sigmoid中取4舍5入(大于百分之五十才算1)，得出eq = tensor([False, False, False,  True,  True...])
            equal = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            # 先进行真值转数值的处理，之后得出正确率，例如2/5=0.4
            accuracy = torch.mean(equal.float())

        total_loss += output.item()
        total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'acc': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()          # 开始进行一个epoch结束的验证
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():           # 消除梯度
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model_train(images)
            output = loss(outputs, targets)

            equal = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy = torch.mean(equal.float())
        # 总损失和总正确率
        val_loss += output.item()
        val_total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'acc': val_total_accuracy / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        # 记录到loss列表中的是每个batch的平均损失（loss是一个自定义的类，有loss属性）
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        # 打印平均损失
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:                # 每个周期保存一次权重
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        # 最优权重计算方式---只看测试集的效果判断最优，val_loss是一个每次平均loss列表，如果只训练了一次或平均损失更小，就保存最有权重
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch%03d-loss%.3f-val_loss%.3f.pth" %
                                                        (epoch+1, total_loss/epoch_step, val_loss/epoch_step_val)))
        # 记录最优秀是测试集损失最低的那一趟
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
