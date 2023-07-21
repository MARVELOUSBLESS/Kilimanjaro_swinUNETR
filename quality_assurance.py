# libraries to load nifti images and save them
import os
import  nibabel as nib
import numpy as np
import matplotlib
from glob import glob

# libararies to load the model:
import torch
from monai.networks.nets import SwinUNETR

# libraries to call another python scritp 
import subprocess
import sys

def _test_save_image_png():
    # to test save_image_png
    path_2_img = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00387-000/BraTS-GLI-00387-000-seg.nii.gz"
    outname = './qa_output/'+os.path.basename(path_2_img).split(".")[0]+".png"
    # print(outname)
    image=load_1_nifti(path_2_img)
    save_image_png(image, 45, outname)


def save_image_png(img:np.array,slice_number:int,outname:str):
    r'''given a 3D numpy array, it will save a slice as a png image'''
    # Access the image data and display it
    slice = img[:, :,img.shape[2] // 2]
    slice = img[:, :,slice_number]
    matplotlib.image.imsave(outname, slice)
    # matplotlib.image.imsave('./qa_output/'+os.path.basename(path_2_img).split(".")[0]+".png", slice)

def load_1_nifti(path_2_img):
    # Load the image using nibabel
    return nib.load(path_2_img).get_fdata()

def load_model(path_2_model:str, model_shape:dict):
    r""" NOT NEEDED use test.py to generate model prediction as nifti files. 
    
    Given the path to a Swin UNETR model that has been previously trained, this function generates a new model 
        and transfers the state of the trained model to the new model. It then returns the new model.

    inputs:
        - path_2_model := path to the .pt file
        - model_shape := a dictionary containing the following keys and values
            - "in_channels":int := # input channels
            - "out_channels":int := # output channels
            - "feature_size":int := the size of the features. it must be multiple of 12
            - "use_checkopint":bool := to see if they want to use checkpoint or not
    """
    # initialize an empty model
    model = SwinUNETR(
        img_size=128,
        in_channels=model_shape.in_channels,
        out_channels=model_shape.out_channels,
        feature_size=model_shape.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=model_shape.use_checkpoint,
    )
    # load the state dictionary of the trained model
    model_dict = torch.load(path_2_model)["state_dict"]
    # fill up the new model with the loaded state dictionary
    model.load_state_dict(model_dict)
    # put the new model into evaluation mode
    model.eval()

    return model

def _test_load_patient_to_tensor():
    patient_dir = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00387-000/"
    load_patient_to_tensor(patient_dir)

def load_patient_to_tensor(path_2_patient:str):
    r''' Given the folder of a patient, it will load the patient MRI images along with the segmentation into a dictionary of two tensors. 
            The first tensor as the patient scan with the 4 channels the second tensor has the patient label. 
    
    inputs:
        - path_2_patient := the global path to a single patient folder containing the following nifti files:
            - t2w.nii.gz
            - t2f.nii.gz
            - t1n.nii.gz
            - t1c.nii.gz
    returns:
        - patient_dict_tensor := a dictionary containing the following keys and values
            - "input":torch.Tensor :=  a tensor of shape [4, 128, 128, 128] containig the patient mri scans in each channel.  
                    the channels represent the t2w, t2f, t1n, and t1c scans.
            - "label":torch.Tensor := a tensor of shape [3, 128, 128, 128] the ground truth segmentation of the patient tumors.  
                    the channels represent the tumor core, enhancing tumor and tumor volume. 
    '''

    # let's get the global paths to nifti images
    path_2_nifties = glob(path_2_patient+'*.nii.gz')
    # print(path_2_nifties)

    # load each nifti image
    patient_tensors = dict((os.path.basename(img_name).split(".")[0], torch.from_numpy(load_1_nifti(img_name))) for img_name in path_2_nifties)
    # print(patient_tensors.keys())

    label = torch.zeros_like(list(patient_tensors.values())[0].shape)
    input = torch.Tensor()

    for key in patient_tensors:
        if "seg" in key:
            label = patient_tensors[key]
        

def match_prediction_name(dir_predictions:str):

    prediction_dir_list = glob(dir_predictions+'*.nii.gz')

    for old_pred_path in prediction_dir_list:
        os.rename(old_pred_path, dir_predictions+"BraTS-GLI-"+"-".join(os.path.basename(old_pred_path).split(".")[0].split("-")[1:3])+"-seg.nii.gz")

    # patient_number_list = ["BraTS-GLI-"+"-".join(os.path.basename(path_2_prediction).split(".")[0].split("-")[1:3])+"-seg.nii.gz" for path_2_prediction in glob(dir_predictions+'*.nii.gz')]
    # print(patient_number_list)


def main():
    # _test_save_image_png()      # test passed
    # _test_load_patient_to_tensor()

    path_patient="/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00387-000/"
    path_predictions="/home/guest183/research-contributions/SwinUNETR/BRATS21/outputs/4gpu_120_epoch/"
    # path_model='/home/guest183/research-contributions/SwinUNETR/BRATS21/runs/4_gpu_4_epochs/model_final.pt'

    match_prediction_name(path_predictions)
    # screen shot at a specific z index for prediction, ground truth, and all the MRI images
    # subtract prediction from ground truth
    # save the result for that patient



 # DO NOT DELETE
if __name__ == "__main__":

    main()

   
    
  