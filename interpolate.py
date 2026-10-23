import argparse

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import autoencoderMLP4Layer

DATA_DIR = './data'   # change if using Colab

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def main():
    print('running main ...')

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [8]')
    argParser.add_argument('-n', metavar='steps', type=int, default=8, help='number of interpolation steps [8]')

    args = argParser.parse_args()

    save_file = args.s
    bottleneck_size = args.z
    n_steps = args.n

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = FashionMNIST(DATA_DIR, train=False, download=False, transform=test_transform)

    N_input = 28 * 28
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file, weights_only=True))
    model.to(device)
    model.eval()

    print('enter two indices from 0 to', len(test_set) - 1, 'for the two images to interpolate between')

    idx1 = int(input('index 1 > '))
    idx2 = int(input('index 2 > '))

    label1 = test_set.targets[idx1].item()
    label2 = test_set.targets[idx2].item()
    print('image 1: label =', label1, '(' + CLASSES[label1] + ')')
    print('image 2: label =', label2, '(' + CLASSES[label2] + ')')

    # normalize and flatten, same as test.py
    img1 = test_set.data[idx1].type(torch.float32) / 255.0
    img1 = img1.view(1, N_input).to(device=device)

    img2 = test_set.data[idx2].type(torch.float32) / 255.0
    img2 = img2.view(1, N_input).to(device=device)

    with torch.no_grad():
        z1 = model.encode(img1)
        z2 = model.encode(img2)

        # n interior alpha values, strictly between 0 and 1
        alphas = torch.linspace(0, 1, n_steps + 2)[1:-1]

        interpolated_outputs = []
        for a in alphas:
            z = (1 - a) * z1 + a * z2
            out = model.decode(z)
            interpolated_outputs.append(out)

        # decode the two original endpoints sprately
        out1 = model.decode(z1)
        out2 = model.decode(z2)

    # full row: original1, n genuinely-interpolated steps, original2 → n + 2 total, no duplicates
    all_outputs = [out1]  + interpolated_outputs + [out2]

    fig, axes = plt.subplots(1, n_steps + 2, figsize=(2 * (n_steps + 2), 2))
    for i, out in enumerate(all_outputs):
        img = out.view(28, 28).cpu()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    axes[0].set_title(CLASSES[label1], fontsize=8)
    axes[-1].set_title(CLASSES[label2], fontsize=8)

    plt.tight_layout()
    plt.savefig('interpolation.png')
    print('saved interpolation.png')
    plt.show()


if __name__ == '__main__':
    main()