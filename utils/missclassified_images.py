import matplotlib.pyplot as plt
import numpy as np
import torch

def misclassified_images(model, device, test_data_loader, list_of_classes):
    list_of_misclassified_images = []

    for images, labels in test_data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            if (predicted[i] != labels[i]):
                list_of_misclassified_images.append([images[i], predicted[i], labels[i]])
        if (len(list_of_misclassified_images) > 25):
            break

    fig = plt.figure(figsize=(8, 8))
    for i in range(20):
        sub = fig.add_subplot(5, 5, i + 1)
        # Loading image from cpu to gpu
        image = list_of_misclassified_images[i][0].cpu()
        # Restoring image from normalization
        image = image / 2 + 0.5
        image = image.numpy()
        plt.imshow(np.transpose(image, (1, 2, 0)), interpolation='none')

        sub.set_title("Predicted={},\n Target={}".format(str(list_of_classes[list_of_misclassified_images[i][1].data.cpu().numpy()]),
                                          str(list_of_classes[list_of_misclassified_images[i][2].data.cpu().numpy()])))

    plt.tight_layout(pad=3.0)
    return list_of_misclassified_images