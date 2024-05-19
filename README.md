## Getting Started

### Overview
A Convolutional Siamese network is implemented, employing the Contrastive loss function as described in the reference algorithm 'SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification'. The Euclidean distance is selected as the metric for comparing the output feature vectors to compute accuracy. Euclidean distance and SSIM are used to display the similarity percentage during inference.

### Install dependencies
#### Requirements
- PyTorch>=1.4.0
- numpy==1.18.1
- scipy==1.4.1
- scikit-learn==0.22.1
- flask==1.1.1
- gunicorn
- pillow==7.1.2
- gdown==4.7.3

```
pip install -r requirements.txt
```
### Dataset
The CEDAR signature dataset is used to train the network. The CEDAR dataset is one of the benchmark datasets for signature verification. It consists of 24 genuine and forged signatures each from 55 different signers.

[Dataset link](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar)

### Train instructions

Set the downloaded CEDAR dataset paths in "Dataloaders.py"

Run train.py to get the Convolutional Siamese network trained on the CEDAR dataset

```
python train.py
```
The resultant trained model weights will be saved to `Models/` by default

### Test instruction using pretrained model
```
python test_image_match.py --image1 --image2 --checkpoint --threshold
```
### Arguments
* `--image1`: real image to compare with
* `--image2`: test image fr verification
* `--checkpoint`: pretrained siamese model
* `--threshold`: confidence threshold

### Acknowledgements
Reference algorithm is SigNet: Convolutional Siamese Network for Writer Independent Offline SignatureVerification
This implementation has been based on the repository https://github.com/Aftaab99/OfflineSignatureVerification
