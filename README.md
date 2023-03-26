# LAB1
Detect Pneumonia from chest X-ray images

## Import
import matplotlib.pyplot, pytorch, torchvision, seaborn, etc.

## Measurement
Output : tp, tn, fp, fn

## Train
Output : train_acc_list, val_acc_list, f1_score_list, best_c_matrix

## Test
Output : val_acc, f1_score, c_matrix

## Model
利用ResNet50 (50 layers)中的pretrained model作為基礎，對模型進行訓練。

Num_epochs = 30

Batch_size = 64

Learning_rate (lr) = 1e-5

Weight_decay (wd) = 0.01

Num_classes = 2

## Data augmentation
Degree = 90

Resize = 224

## Loss function, Optimizer
nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))

## Plot 
Training and testing accuracy curve

Testing f1 score curve
 
Confusion matrix
