import torchvision.transforms as T


# These functions transform the input images for training and evaluation
# For training images are transformed with randomresize and randomflips to ensure a better generalization of the model.
# Use these functions to transform the input images for training and evaluation

def get_train_transforms():
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


def get_eval_transforms():
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])