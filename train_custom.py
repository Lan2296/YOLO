
import mlflow.pytorch
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

import numpy as np 

from custom_model import YOLO,  ANCHORS, test_transform, s, CONF_THRESHOLD, train_transform
from HelpFunction import  device, check_class_accuracy, convert_cells_to_bboxes, plot_image, nms
from Dataset import Dataset
from YoloLoss import YOLOLoss

import mlflow
import mlflow.pytorch
from PIL import Image

# Hyper Parameter
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.5e-4
NUM_CLASS = 26
target_size=(832,832)

print(f"Training device: {device}")
#Training loop
# Define the train function to train the model 
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch): 
    model.train()
    # Creating a progress bar 
    progress_bar = tqdm(loader, leave=True) 

    # Initializing a list to store the losses 
    losses = [] 

    # Iterating over the training data 
    for _, (x, y) in enumerate(progress_bar): 
        x = x.to(device) 
        y0, y1, y2 = ( 
            y[0].to(device),
            y[1].to(device), 
            y[2].to(device), 
        ) 

        with torch.cuda.amp.autocast(): 
            # Getting the model predictions 
            outputs = model(x) 
            # Calculating the loss at each scale 
            loss = ( 
                loss_fn(outputs[0], y0, scaled_anchors[0]) 
                + loss_fn(outputs[1], y1, scaled_anchors[1]) 
                + loss_fn(outputs[2], y2, scaled_anchors[2]) 
            ) 

        # Add the loss to the list 
        losses.append(loss.item()) 

        # Reset gradients 
        optimizer.zero_grad() 

        # Backpropagate the loss 
        scaler.scale(loss).backward() 

        # Optimization step 
        scaler.step(optimizer) 

        # Update the scaler for next iteration 
        scaler.update() 

        # update progress bar with loss 
        mean_loss = sum(losses) / len(losses) 
        progress_bar.set_postfix(loss=mean_loss)

        #mlflow.log_metric("Batch loss", loss.item(), step= _)

        mlflow.log_metric("epoch mean loss: ", mean_loss, step=epoch)
       
       


def evaluate(dataloader, model, loss_fn, scaled_anchors):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    val_loss =  0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x,y in dataloader:
            x = x.to(device)
            y0, y1, y2 = ( 
            y[0].to(device),
            y[1].to(device), 
            y[2].to(device), 
        )            
            with torch.cuda.amp.autocast(): 
            # Getting the model predictions 
                pred= model(x) 
                # Calculating the loss at each scale 
                loss = ( 
                    loss_fn(pred[0], y0, scaled_anchors[0]) 
                    + loss_fn(pred[1], y1, scaled_anchors[1]) 
                    + loss_fn(pred[2], y2, scaled_anchors[2]) 
                ) 
                val_loss += loss.item()          
    val_loss /= num_batches

    return val_loss
# Creating  dataset object 
'''
train_dataset = Dataset( 
    yaml_file = "/app/yolo/Dataset/dataset2/custom_data2.yaml",
    image_dir = "/app/yolo/Dataset/dataset2/train/images",
    label_dir = "/app/yolo/Dataset/dataset2/train/labels",
    grid_sizes=[13, 26, 52], 
    anchors=ANCHORS, 
    transform=train_transform 
) 
'''
train_dataset = Dataset( 
    yaml_file = "/app/yolo/Dataset/Neu_Dataset/custom_data2.yaml",
    image_dir = "/app/yolo/Dataset/Neu_Dataset/train/images",
    label_dir = "/app/yolo/Dataset/Neu_Dataset/train/labels",
    grid_sizes=[13, 26, 52], 
    anchors=ANCHORS, 
    transform=train_transform 
) 
'''
val_dataset = Dataset( 
        yaml_file = "/app/yolo/Dataset/dataset2/custom_data2.yaml",
        image_dir = "/app/yolo/Dataset/dataset2/valid/images",
        label_dir =  "/app/yolo/Dataset/dataset2/valid/labels",
        grid_sizes=[13, 26, 52], 
        anchors=ANCHORS, 
        transform=test_transform 
    ) 
'''
val_dataset = Dataset( 
    yaml_file = "/app/yolo/Dataset/Neu_Dataset/custom_data2.yaml",
    image_dir = "/app/yolo/Dataset/Neu_Dataset/valid/images",
    label_dir = "/app/yolo/Dataset/Neu_Dataset/valid/labels",
    grid_sizes=[13, 26, 52], 
    anchors=ANCHORS, 
    transform=test_transform 
) 
test_dataset = Dataset( 
        yaml_file = "/app/yolo/Dataset/dataset2/custom_data2.yaml",
        image_dir = "/app/yolo/Dataset/dataset2/test/images",
        label_dir = "/app/yolo/Dataset/dataset2/test/labels",
        grid_sizes=[13, 26, 52], 
        anchors=ANCHORS, 
        transform=test_transform 
    ) 
     
# Creating  dataloader object 
train_loader = DataLoader( 
    train_dataset, 
    batch_size= BATCH_SIZE, 
    #num_workers = 0,
    shuffle=True, 
    pin_memory = True,
        
    ) 
test_loader = DataLoader( 
        test_dataset, 
        batch_size= BATCH_SIZE, 
        num_workers = 0,
        shuffle=True, 
        pin_memory = True,
        
    ) 

    # Creating a dataloader object 
val_loader = DataLoader( 
        val_dataset, 
        batch_size= BATCH_SIZE, 
        num_workers = 0,
        shuffle=True, 
        pin_memory = True,
        
    ) 

# Initialize model, optimizer, and loss function
model = YOLO().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = YOLOLoss() # Your loss function from earlier
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("YOLO Training")

def main ():
    best_loss = float('inf')
    epoch_no_improve = 0
    #patience=5
  
    # Defining the scaler for mixed precision training 
    scaler = torch.cuda.amp.GradScaler()
    # Scaling the anchors 
    scaled_anchors = ( 
        torch.tensor(ANCHORS) * 
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
    ).to(device) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode ='min', factor =0.1, patience=3)
    mlflow.pytorch.autolog()

    with mlflow.start_run():


        mlflow.log_params({"epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "optimizer": optimizer,})

        for epoch in range(EPOCHS): 
        
            print(f"Epoch: {epoch+1}/{EPOCHS}, Learning Rate: {optimizer.param_groups[0]['lr']} ") 
            training_loop(train_loader, model, optimizer, criterion, scaler, scaled_anchors, epoch=epoch)
            check_class_accuracy(model=model, loader= train_loader, threshold= CONF_THRESHOLD,epoch= epoch)
            val_loss = evaluate(val_loader, model,  criterion, scaled_anchors)     
            scheduler.step(val_loss) # Learning rate mit valuation loss anpassen um Overfiting zuvermeiden
            mlflow.log_metric("Val loss", val_loss, step=epoch)
    
            # Schleife um die Beste Model speichern und frühe aufhören falls Overfiting gibt
            if val_loss < best_loss:
                best_loss = val_loss
                epoch_no_improve = 0
                # Speichern des bestes Modells lokal
                torch.save(model.state_dict(),"/app/yolo/best_model.pth")
                print(f"Best Model update (val_loss: {val_loss:.4f})")
                # Logge Modell zu Mlflow
                mlflow.pytorch.log_model(model, artifact_path="best_model")
            else:
                epoch_no_improve +=1
                print(f" No improvement for {epoch_no_improve} epoch")
              
            
            

import matplotlib.pyplot as plt
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2


def test():   
       
    model_uri = 'runs:/25ec0f1cb17b4249b308c50b29e18c73/best_model'
    #model_uri = 'runs:/9b0b7712e3904e42b69ec62357fe25c7/best_model'
    #model_uri = 'runs:/5ed02747e2f64c86991a7349187b25d2/best_model'
    loaded_model = mlflow.pytorch.load_model(model_uri).to(device)
    
    loaded_model.eval()

    image = Image.open("/app/yolo/Test Bild/2025-06-03-120848.jpg").convert("RGB")
    #image_resize= image.resize((416,416),resample = 1)
    image_np =np.array(image)
    image_crop = image.crop((int(image_np.shape[1]/2)-int(target_size[1]/2), image_np.shape[0]-target_size[0], int(image_np.shape[1]/2)+int(target_size[1]/2), image_np.shape[0]))
    image_resize= image_crop.resize((416,416),resample = 0)
    image_np =np.array(image_resize)
    #image_croped= image_np[image_np.shape[0]-target_size[0]:image_np.shape[0], int(image_np.shape[1]/2)-int(target_size[1]/2):int(image_np.shape[1]/2)+int(target_size[1]/2)]
    
    transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=416), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=416, min_width= 416, border_mode=cv2.BORDER_CONSTANT
		), 
        # Normalize the image 
		A.Normalize( 
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
			
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], )
    transformed = transform(image =image_np)
    image_tensor = transformed["image"]
 
    scaled_anchors = ( 
        torch.tensor(ANCHORS) * 
        torch.tensor(s).unsqueeze(1).unsqueeze(2)#.repeat(1,3,2) 
    ).to(device) 
    image_tensor = image_tensor.unsqueeze(0).to(device)
    #loaded_model.to(image_tensor, device)
    with torch.no_grad():
        outputs = loaded_model(image_tensor)
        
        boxes =[]
        for i, out in enumerate(outputs):
            grid_size= out.shape[2]
            anchor_set = scaled_anchors[i]

            converted = convert_cells_to_bboxes(predictions=out, anchors=anchor_set, s=grid_size, is_predictions=True)
            boxes.append(converted)

        all_boxes = [b for scale in boxes for b in scale]
        boxes_for_image = all_boxes[0]
        boxes_for_image =[box for box in boxes_for_image if box[1]>0.5]
        #nms_boxes = nms(bboxes=boxes_for_image, iou_threshold=0.5,threshold=0.3)
        nms_boxes = nms(bboxes=boxes_for_image, iou_threshold=0.5,threshold=0.5)

        plot_image(image_np, nms_boxes)
        


if __name__ == "__main__":
    #main()
    test()
 
    