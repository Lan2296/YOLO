import torch

from torch.utils.data import DataLoader, Dataset

from PIL import Image
import os
import numpy as np
import yaml
from glob import glob
from HelpFunction import iou

# Create a dataset class to load the images and labels from the folder 
class Dataset(torch.utils.data.Dataset): 
    def __init__( 
        self, yaml_file, image_dir, label_dir, anchors, 
        image_size=416, grid_sizes=[13, 26, 52], 
        num_classes=26, transform=None
    ): 
        # Read the csv file with image names and labels 
        with open ("/app/yolo/Dataset/dataset2/custom_data2.yaml") as f:
            data = yaml.safe_load_all(f) ##
            loaded_data = list(data)
    
        # Pfad aus Yaml laden
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        # Bilddateien sammeln
        self.image_files = glob(os.path.join(self.image_dir,'*.jpg')) + glob(os.path.join(self.image_dir,'*.png'))
        
        # Modellparameter
        # Image size 
        self.image_size = image_size 
        # Transformations 
        self.transform = transform 
        # Grid sizes for each scale 
        self.grid_sizes = grid_sizes 
        # Anchor boxes 
        self.anchors = torch.tensor( 
            anchors[0] + anchors[1] + anchors[2]) 
        # Number of anchor boxes 
        self.num_anchors = self.anchors.shape[0] 
        # Number of anchor boxes per scale 
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes 
        self.num_classes = num_classes 
        # Ignore IoU threshold 
        self.ignore_iou_thresh = 0.5

    def __len__(self): 
        return len(self.image_files) 
    
    def __getitem__(self, idx): 
        # Getting the label path and Bild path
        image_path= self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # label_path = os.path.join(self.label_dir, self.image_files.iloc[idx, 1]) 
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        
        # We are applying roll to move class label to the last column 
        # 5 columns: x, y, width, height, class_label 
        # Label einlesen
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
    
            
        
        # Bild laden
        #img_path = os.path.join(self.image_dir, self.image_files.iloc[idx, 0]) 
        image = np.array(Image.open(image_path).convert("RGB")) 

        # Albumentations augmentations 
        if self.transform: 
            augs = self.transform(image=image, bboxes=bboxes) 
            image = augs["image"] 
            bboxes = augs["bboxes"] 

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
        # target : [probabilities, x, y, width, height, class_label] 
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
                for s in self.grid_sizes] 
        
        # Identify anchor box and cell for each bounding box 
        for box in bboxes: 
            # Calculate iou of bounding box with anchor boxes 
            iou_anchors = iou(torch.tensor(box[2:4]), 
                            self.anchors, 
                            is_pred=False) 
            # Selecting the best anchor box 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
            x, y, width, height, class_label = box 

            # At each scale, assigning the bounding box to the 
            # best matching anchor box 
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices: 
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
                
                # Identifying the grid size for the scale 
                s = self.grid_sizes[scale_idx] 
                
                # Identifying the cell to which the bounding box belongs 
                i, j = int(s * y), int(s * x) 
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                
                # Check if the anchor box is already assigned 
                if not anchor_taken and not has_anchor[scale_idx]: 

                    # Set the probability to 1 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative 
                    # to the cell 
                    x_cell, y_cell = s * x - j, s * y - i 

                    # Calculating the width and height of the bounding box 
                    # relative to the cell 
                    width_cell, height_cell = (width * s, height * s) 

                    # Idnetify the box coordinates 
                    box_coordinates = torch.tensor( 
                                        [x_cell, y_cell, width_cell, 
                                        height_cell] 
                                    ) 

                    # Assigning the box coordinates to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 

                    # Assigning the class label to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 

                    # Set the anchor box as assigned for the scale 
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the 
                # IoU is greater than the threshold 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
                    # Set the probability to -1 to ignore the anchor box 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target 
        return image, tuple(targets)
    
from HelpFunction import convert_cells_to_bboxes, iou, nms, plot_image
from custom_model import ANCHORS, test_transform, class_labels
def test():
    anchors = ANCHORS

    transform = test_transform

    dataset = Dataset(
        yaml_file = "/app/yolo/Dataset/dataset2/custom_data2.yaml",
        image_dir = "/app/yolo/Dataset/dataset2/test/images",
        label_dir = "/app/yolo/Dataset/dataset2/test/labels",
        anchors=anchors,
        transform=transform,
    )
    s = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    print("type:", type(class_labels))
    print("lenght des class labels:", len(class_labels))
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += convert_cells_to_bboxes(
                y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7)
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()