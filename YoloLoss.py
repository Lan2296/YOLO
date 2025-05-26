import torch
import torch.nn as nn 

from HelpFunction import iou 

# Defining YOLO loss class 
class YOLOLoss(nn.Module): 
	def __init__(self): 
		super().__init__() 
		self.mse = nn.MSELoss() 
		self.bce = nn.BCEWithLogitsLoss() 
		self.cross_entropy = nn.CrossEntropyLoss() 
		self.sigmoid = nn.Sigmoid() 
	
	def forward(self, pred, target, anchors): 
		# Identifying which cells in target have objects 
		# and which have no objects 
		obj = target[..., 0] == 1
		no_obj = target[..., 0] == 0

		# Calculating No object loss 
		no_object_loss = self.bce( 
			(pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]), 
		) 

		# Calculating Objekt loss
		# Reshaping anchors to match predictions 
		anchors = anchors.reshape(1, 3, 1, 1, 2) 
		# Box prediction confidence 
		box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), 
							torch.exp(pred[..., 3:5]) * anchors 
							],dim=-1) 
		# Calculating intersection over union for prediction and target 
		ious = iou(box_preds[obj], target[..., 1:5][obj]).detach() 
		# Calculating Object loss 
		object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj]) 

		
		# Predicted box coordinates: x,y
		pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) 
		# Target box coordinates 
		target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors) 
		# Calculating box coordinate loss: width, height
		box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj]) 

		
		# Calculating class loss 
		class_loss = self.cross_entropy((pred[..., 5:][obj]), target[..., 5][obj].long()) 

		# Total loss 
		return ( 
			box_loss 
			+ object_loss 
			+ no_object_loss 
			+ class_loss 
		)
  