from Preprocessing import invert_image_path, convert_to_image_tensor
from Model import SiameseConvNet, distance_metric, ssim
from torch import load
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))


def plot(im1,im2,text=None):
  # create figure
  fig = plt.figure(figsize=(10, 7))
  
  # setting values to rows and column variables
  rows = 1
  columns = 2
    
  # reading images
  Image1 = cv2.imread(im1)
  Image2 = cv2.imread(im2)
  
  # Adds a subplot at the 1st position
  fig.add_subplot(rows, columns, 1)
  # showing image
    
  plt.imshow(Image1)
  plt.axis('off')
  plt.title("Real")
  
  # Adds a subplot at the 2nd position
  fig.add_subplot(rows, columns, 2)
  # showing image
  plt.imshow(Image2)
  plt.axis('off')
  plt.title("Test image Label: {}".format(text))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--image1', type=str, default="path/to/original_image.png")
    parser.add_argument('--image2', type=str, default="path/to/test_image.png")
    parser.add_argument('--checkpoint', type=str, default='path/to/Models/weights')
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    real = convert_to_image_tensor(invert_image_path(args.image1))
    sample = convert_to_image_tensor(invert_image_path(args.image2))
    concatenated = torch.cat((real,sample),0)

    model = SiameseConvNet()
    model.load_state_dict(load(open(args.checkpoint, 'rb'), map_location=device))

    print('Verifying:')
    # getting feature embeddings from siamese network
    f_a, f_b = model.forward(real, sample)

    dist = distance_metric(f_a, f_b)
    dist = dist.detach().numpy()
    similarity = ssim(f_a, f_b)

    print("MSE: {}, structural_similarity: {}".format(dist, similarity))
    prediction = np.where(dist <= args.threshold, 1, 0)
    L = ['Forged', 'Real']
    label = L[prediction[0]]

    plot(args.image1, args.image2, text=label)
    plt.show()

