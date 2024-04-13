import os
import time
import numpy as np
from glob import glob
import cv2
import torch
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from model import build_unet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    iou = intersection / (union + 1e-7)  # avoid division by zero

    return iou

def calculate_dice_score(y_true, y_pred):
    """Calculate Dice Score"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    dice_score = (2. * intersection + 1e-7) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)  # avoid division by zero
    dice_score = np.clip(dice_score, 0, 1)  # Clip values to [0, 1]
    return dice_score

def mask_parse(mask):
    """Convert mask to RGB"""
    mask = np.expand_dims(mask, axis=-1)  # (256, 256, 1)
    mask = np.concatenate([mask] * 3, axis=-1)  # (256, 256, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Create result directory """
    create_dir("test_pred")

    """ Load dataset """
    test_x = sorted(glob("/home/fteam5/strain/py_script/ghj/test/ugly/*"))
    test_y = sorted(glob("/home/fteam5/strain/py_script/ghj/test/ugly_mask/*"))

    """ Hyperparameters """
    H = 256
    W = 256
    size = (W, H)
    checkpoint_path = "/home/fteam5/strain/checkpoint/check.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_iou = 0
    total_dice_score = 0
    num_samples = len(test_x)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=num_samples):
        """ Extract the name """
        name = os.path.basename(x).split(".")[0]

        """ Reading image """
        test_image = cv2.imread(x, cv2.IMREAD_COLOR)
        test_image = cv2.resize(test_image, (256, 256))

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))

        """ Normalization """
        test_image = test_image / 255.0
        test_image = np.transpose(test_image, (2, 0, 1))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image.astype(np.float32)
        test_image = torch.from_numpy(test_image).to(device)

        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask).to(device)

        try:
            with torch.no_grad():
                """ Prediction """
                pred_mask = model(test_image)
                pred_mask = torch.sigmoid(pred_mask)

                pred_mask = pred_mask[0].cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

                """ Saving predicted mask """
                _, pred_mask_binary = cv2.threshold(pred_mask, 0.5, 255, cv2.THRESH_BINARY)
                cv2.imwrite(f"/home/fteam5/strain/py_script/ghj/result_test/ugly_pred/{name}_pred_mask.png", pred_mask_binary.squeeze())

                """ Calculate IoU """
                iou = calculate_metrics(mask.cpu().numpy(), pred_mask_binary)
                total_iou += iou

                """ Calculate Dice Score """
                dice_score = calculate_dice_score(mask.cpu().numpy(), pred_mask_binary)
                total_dice_score += dice_score

                """ Print IoU and Dice Score """
                print(f"IoU for {name}: {iou}")
                print(f"Dice Score for {name}: {dice_score}")

        except Exception as e:
            print(f"Error occurred in processing {name}: {e}")

    """ Calculate average IoU and Dice Score """
    average_iou = total_iou / num_samples
    average_dice_score = total_dice_score / num_samples
    print(f"Average IoU: {average_iou}")
    print(f"Average Dice Score: {average_dice_score}")
