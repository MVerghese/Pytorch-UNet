import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet_utils.data_loading import BasicDataset
from unet import UNet

from matplotlib import pyplot as plt
import cv2



class Masker:
	def __init__(self, model_path):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = UNet(n_channels=3, n_classes=2, bilinear=False)
		self.model.to(self.device)
		state_dict = torch.load(model_path, map_location=self.device)
		mask_values = state_dict.pop('mask_values', [0, 1])
		self.model.load_state_dict(state_dict)
		self.model.eval()

	def predict(self,img):
		# convert image to PIL
		if type(img) == np.ndarray:
			img = Image.fromarray(img)
		elif type(img) == torch.Tensor:
			img = transforms.ToPILImage()(img)

		# predict mask
		torch_img = torch.from_numpy(BasicDataset.preprocess(None, img, 1, is_mask=False))
		torch_img = torch_img.unsqueeze(0)
		torch_img = torch_img.to(device=self.device, dtype=torch.float32)
		with torch.no_grad():
			output = self.model(torch_img).cpu()
			output = F.interpolate(output, (img.size[1], img.size[0]), mode='bilinear')
			if self.model.n_classes > 1:
				mask = output.argmax(dim=1)
			else:
				mask = torch.sigmoid(output) > out_threshold
		np_mask = mask[0].long().squeeze().numpy()

		return np_mask

def test_masker():
	masker = Masker("/media/mverghese/Mass Storage/models/robot_mask_unet/checkpoint_epoch100.pth")
	img_path = "/home/mverghese/MBLearning/RLBench_frames/0000123456.png"
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	mask = masker.predict(img)
	plt.imshow(img)
	plt.show()
	plt.imshow(mask)
	plt.show()

if __name__ == "__main__":
	test_masker()




