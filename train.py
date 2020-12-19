from models import Detector, save_model
from utils import accuracy, load_data, ConfusionMatrix,LABEL_NAMES
import torch,torchvision
import numpy as np
from os import path
from torchvision import transforms
from torchvision.transforms import functional as F

import torch.utils.tensorboard as tb

TRAIN_PATH = "data/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    train_data = load_data(TRAIN_PATH,batch_size=256)
    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    import inspect
    transform = eval(args.transform,
                        {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        confusion = ConfusionMatrix(len(LABEL_NAMES))
        for img, label in train_data:
            if train_logger is not None:
                train_logger.add_images('augmented_image', img[:4])
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            confusion.add(logit.argmax(1), label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('accuracy', confusion.global_accuracy, global_step)
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.imshow(confusion.per_class, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(confusion.per_class.size(0)):
                for j in range(confusion.per_class.size(1)):
                    ax.text(j, i, format(confusion.per_class[i, j], '.2f'),
                            ha="center", va="center", color="black")
            train_logger.add_figure('confusion', f, global_step)

        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=75)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)


