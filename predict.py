import argparse
import torch
import json
from torchvision import models, transforms
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {checkpoint['arch']}")

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pil_image = Image.open(image)
    img_tensor = preprocess(pil_image)
    np_image = img_tensor.numpy()

    return np_image

def predict(image_path, model, topk, category_names, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device, dtype=torch.float)

    with torch.no_grad():
        output = model.forward(img_tensor)

    probabilities, classes = torch.exp(output).topk(min(topk, len(model.class_to_idx)))

    class_to_idx = model.class_to_idx

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f, strict=False)

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        classes = [cat_to_name[idx_to_class[i]] for i in classes]

    for i in range(len(probabilities)):
        print(f"Prediction {i+1}: {classes[i]} ({cat_to_name[classes[i]]}) with probability {probabilities[i]:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained deep learning model')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)

    predict(args.image_path, model, args.top_k, args.category_names, args.gpu)

if __name__ == '__main__':
    main()