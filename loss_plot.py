import os
import re
import matplotlib.pyplot as plt

file_path = "fourier_series_ai-42301.out"

if not os.path.exists(file_path):
    raise FileNotFoundError(file_path)

file_aspect_ratio = 137/90

EPOCH_RE = re.compile(r'^EPOCH (\d+):')
LOSS_RE = re.compile(r'^LOSS train ([\d.eE+-]+) valid ([\d.eE+-]+)')


def get_data():
    source = open(file_path)
    train_loss_list = []
    test_loss_list = []
    epoch_loss_list = []
    # LOSS train 0.2917159233391285 valid 0.305107751917094

    content = source.read().split("\n")

    current_epoch = None
    for i in range(len(content)):
        epoch_match = EPOCH_RE.match(content[i])
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        loss_match = LOSS_RE.match(content[i])
        if loss_match:
            train_loss_list.append(float(loss_match.group(1)))
            test_loss_list.append(float(loss_match.group(2)))
            # The "EPOCH N:" header always precedes its "LOSS train/valid" line, so
            # current_epoch is already set by the time we get here; fall back to a
            # running count only if the log was truncated before its first header.
            epoch_loss_list.append(current_epoch if current_epoch is not None else len(train_loss_list))

    return train_loss_list, test_loss_list, epoch_loss_list


train_loss, test_loss, epoch_loss = get_data()


def loss_plot():
    fig, ax = plt.subplots(figsize=(8, 8 / file_aspect_ratio))

    ax.plot(epoch_loss, train_loss, label='Train loss')
    ax.plot(epoch_loss, test_loss, label='Test loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs. Validation Loss')
    ax.legend()

    fig.tight_layout()
    fig.savefig("loss_graph.png")

if __name__ == '__main__':
    loss_plot()
