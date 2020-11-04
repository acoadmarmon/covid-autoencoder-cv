import data_loaders
import matplotlib.pyplot as plt
image_dataset = data_loaders.CTScanDataset(root_dir='./train/')

fig = plt.figure()

for i in range(len(image_dataset)):
    sample = image_dataset[i]

    print(i, sample['image'].shape)

    ax = plt.subplot(1, 2, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['image'])

    if i == 1:
        plt.show()
        plt.savefig('asdf.jpg')
        break