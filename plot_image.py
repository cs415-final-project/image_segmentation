from PIL import Image
import argparse
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('plot image with palette')
parser.add_argument('--image_path', help='path of the image', type=str, default="output/images/img_0_0.png")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

def plot():
    output_image = Image.open(args.image_path)
    output_image = output_image.convert("RGB")
    plt.imsave("output.png", output_image)

if __name__ == '__main__':
    plot()

