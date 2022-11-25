import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def assign_palette():
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")


def eval(model):
    model.eval()
    input_image = Image.open('data/Cityscapes/images/zurich_000014_000019_leftImg8bit.png')
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    plt.imsave("output.png", output_predictions.cpu().numpy())

def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
    eval(model)

if __name__ == '__main__':
    main()