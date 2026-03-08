import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

fig, axes = plt.subplots(2,5, figsize=(10,4))

for i, ax in enumerate(axes.flat):
    img, label = dataset[i]
    ax.imshow(img.permute(1,2,0))
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.show()