import time
import numpy as np
import torch
import pt_util
import MTGnet
import pickle
import torch.optim as optim
from data_fetcher import data_fetcher, test_fetcher


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader, log_interval=None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device=device, dtype=torch.int64)
            output = model(data)
            test_loss_on = model.loss(output, label, reduction='sum').item()
            test_loss += test_loss_on
            pred = output.max(1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    return test_loss, test_accuracy


def network_run(label_dir, is_color, batch_size, learning_rate, weight_decay, epochs, version, name):
    if name == '__main__':
        BATCH_SIZE = batch_size
        TEST_BATCH_SIZE = 15
        EPOCHS = epochs
        LEARNING_RATE = learning_rate
        MOMENTUM = 0.9
        USE_CUDA = True
        SEED = 0
        PRINT_INTERVAL = 10
        WEIGHT_DECAY = weight_decay

        EXPERIMENT_VERSION = version  # increment this to start a new experiment
        LOG_PATH = 'logs/' + EXPERIMENT_VERSION + '/'

        # Now the actual training code
        use_cuda = USE_CUDA and torch.cuda.is_available()

        # torch.manual_seed(SEED)

        device = torch.device("cuda" if use_cuda else "cpu")
        print('Using device', device)
        import torch.multiprocessing as multiprocessing
        print('num cpus:', 4)

        kwargs = {'num_workers': 4,
                  'pin_memory': True} if use_cuda else {}
        class_names, data_train, data_val = data_fetcher(label_dir, 'extra_reduced_images_new', is_color=is_color)
        data_test = test_fetcher(label_dir, 'extra_reduced_images_new', is_color=is_color)

        print(class_names)
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                                   shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=TEST_BATCH_SIZE,
                                                  shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                                  shuffle=False, **kwargs)

        model = MTGnet.MTGNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        start_epoch = model.load_last_model(LOG_PATH)

        train_losses, test_losses, test_accuracies = pt_util.read_log(LOG_PATH + 'log.pkl', ([], [], []))
        val_losses, val_accuracies = [], []

        test_loss, test_accuracy = test(model, device, test_loader)
        val_loss, val_accuracy = test(model, device, val_loader)

        test_losses.append((start_epoch, test_loss))
        test_accuracies.append((start_epoch, test_accuracy))
        val_losses.append((start_epoch, val_loss))
        val_accuracies.append((start_epoch, val_accuracy))
        try:
            for epoch in range(start_epoch, EPOCHS + 1):
                train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
                test_loss, test_accuracy = test(model, device, test_loader)
                val_loss, val_accuracy = test(model, device, val_loader)
                val_losses.append((start_epoch, val_loss))
                val_accuracies.append((start_epoch, val_accuracy))
                train_losses.append((epoch, train_loss))
                test_losses.append((epoch, test_loss))
                test_accuracies.append((epoch, test_accuracy))
                pt_util.write_log(LOG_PATH + 'log.pkl', (train_losses, test_losses, test_accuracies))
                model.save_best_model(test_accuracy, LOG_PATH + '%03d.pt' % epoch)


        except KeyboardInterrupt as ke:
            print('Interrupted')
        except:
            import traceback
            traceback.print_exc()
        finally:
            ep, val = zip(*train_losses)
            pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
            ep, val = zip(*test_losses)
            pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
            ep, val = zip(*test_accuracies)
            pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')
            ep, val = zip(*val_accuracies)
            return max(val)


network_run('monocolor_new', True, 88, 0.020711, 0.001108, 15, '8.1', __name__)