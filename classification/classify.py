import torch
from torchvision import models, transforms
import os
from PIL import Image


def load_image(image_path):
    assert os.path.isfile(image_path)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = data_transform(image)
    image = image.unsqueeze(0)

    return image


def load_model():
    model = models.vgg16(pretrained=True)
    return model


def import_category_dict(category_file):
    file = open(category_file, 'r')
    lines = file.readlines()
    category_dict = {}
    for idx, line in enumerate(lines):
        a = line.strip()
        b = a.split(',')[0]
        category_dict[idx] = b[len(b.split(' ')[0]) + 1:]
    return category_dict


def classify(model, image, category_dict, use_gpu=True, num=5):
    if use_gpu:
        model.cuda()
        image = image.cuda()
    result = model(image)
    result = torch.nn.Softmax(dim=1)(result)
    result = result.squeeze()
    prob, idx = result.sort(descending=True)
    prob, idx = prob[:num].cpu().data.numpy(), idx[:num].cpu().data.numpy()
    pred = [category_dict[i] for i in idx]
    return pred, prob


if __name__ == '__main__':
    image = load_image('../static/images/test.jpg')
    model = load_model()
    category_dict = import_category_dict('../static/images/categories.txt')
    pred, prob = classify(model, image, category_dict, use_gpu=True)
    print(pred)
    print(prob)

