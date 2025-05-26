# Yolo Modell aufbauen
# Importierung der erfordelichen Packete

import torch 
import torch.nn as nn 
  
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 

# Defining CNN Block 
class CNNBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
		super().__init__() 
		self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 
		self.use_batch_norm = use_batch_norm 

	def forward(self, x): 
		# Applying convolution 
		x = self.conv(x) # expected input: Tensor input, Tensor weight, Tensor bias, tuple of ints stride, tuple of ints padding, tuple of ints dilation, int groups)
		# Applying BatchNorm and activation if needed 
		if self.use_batch_norm: 
			x = self.bn(x) 
			return self.activation(x) 
		else: 
			return x

# Defining residual block 
class ResidualBlock(nn.Module): 
	def __init__(self, channels, use_residual=True, num_repeats=1): 
		super().__init__() 
		
		# Defining all the layers in a list and adding them based on number of 
		# repeats mentioned in the design 
		res_layers = [] 
		for _ in range(num_repeats): 
			res_layers += [ 
				nn.Sequential( 
					nn.Conv2d(channels, channels // 2, kernel_size=1), 
					nn.BatchNorm2d(channels // 2), 
					nn.LeakyReLU(0.1), 
					nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
					nn.BatchNorm2d(channels), 
					nn.LeakyReLU(0.1) 
				) 
			] 
		self.layers = nn.ModuleList(res_layers) 
		self.use_residual = use_residual 
		self.num_repeats = num_repeats 
	
	# Defining forward pass 
	def forward(self, x): 
		for layer in self.layers: 
			residual = x 
			x = layer(x) 
			if self.use_residual: 
				x = x + residual 
		return x

# Defining scale prediction class 
class ScalePrediction(nn.Module): 
	def __init__(self, in_channels, num_classes): 
		super().__init__() 
		# Defining the layers in the network 
		self.pred = nn.Sequential( 
			nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
			nn.BatchNorm2d(2*in_channels), 
			nn.LeakyReLU(0.1), 
			nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1), 
		) 
		self.num_classes = num_classes 
	
	# Defining the forward pass and reshaping the output to the desired output 
	# format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
	def forward(self, x): 
		output = self.pred(x) 
		output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
		output = output.permute(0, 1, 3, 4, 2) 
		return output


# Gesammte YOlOv3 model  
class YOLO(nn.Module): 
	def __init__(self, in_channels=3, num_classes=26): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

		# Layers list for YOLOv3 
		self.layers = nn.ModuleList([ 
			CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1), 
			CNNBlock(32, 64, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(64, num_repeats=1), 
			CNNBlock(64, 128, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(128, num_repeats=2), 
			CNNBlock(128, 256, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(256, num_repeats=8), 
			CNNBlock(256, 512, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(512, num_repeats=8), 
			CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(1024, num_repeats=4), ################ endet Darknet-53
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
			CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(1024, use_residual=False, num_repeats=1), 
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(512, num_classes=num_classes), 
			CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
			nn.Upsample(scale_factor=2), 
			CNNBlock(768, 256, kernel_size=1, stride=1, padding=0), 
			CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(512, use_residual=False, num_repeats=1), 
			CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(256, num_classes=num_classes), 
			CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
			nn.Upsample(scale_factor=2), 
			CNNBlock(384, 128, kernel_size=1, stride=1, padding=0), 
			CNNBlock(128, 256, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(256, use_residual=False, num_repeats=1), 
			CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(128, num_classes=num_classes) 
		]) 
	
	# Forward pass for YOLOv3 with route connections and scale predictions 
	def forward(self, x): 
		outputs = [] 
		route_connections = [] 

		for layer in self.layers: 
			if isinstance(layer, ScalePrediction): 
				outputs.append(layer(x)) 
				continue
			x = layer(x) 

			if isinstance(layer, ResidualBlock) and layer.num_repeats == 8: 
				route_connections.append(x) 
			
			elif isinstance(layer, nn.Upsample): 
				x = torch.cat([x, route_connections[-1]], dim=1) 
				route_connections.pop() 
		return outputs
	

##################################### Config ###############################

# Load and save model variable 
load_model = False
save_model = True

# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 
# Default Werte für confident score, mean Average Precision und non maximum suppression
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
# Image size 
image_size = 416

# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 

# Class labels 
class_labels = ['A', 'B', 'C', 'D', 'E', 'F','G','H','I','J','K','M','L','N','O','P','Q','R','S','T','U','V','W','Y','X','Z']

#####################################################################################
# Transform for training 
train_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Random color jittering 
		A.ColorJitter( 
			brightness=0.5, contrast=0.5, 
			saturation=0.5, hue=0.5, p=0.5
		), 
		# Flip the image horizontally 
		A.HorizontalFlip(p=0.5), 
		# Normalize the image 
		A.Normalize( 
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
		
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
		
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams( 
					format="yolo", 
					min_visibility=0.4, 
					label_fields=[] 
				) 
) 

# Transform for testing 
test_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Normalize the image 
		A.Normalize( 
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams( 
					format="yolo", 
					min_visibility=0.4, 
					label_fields=[] 
				) 
)


if __name__ == "__main__":
    num_classes = 26
    IMAGE_SIZE = 416
    model = YOLO(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")