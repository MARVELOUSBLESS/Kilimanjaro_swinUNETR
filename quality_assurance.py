# libraries to load nifti images and save them
import os
import  nibabel as nib
import numpy as np
import matplotlib
from glob import glob
import pandas as pd

# libararies to load the model:
import torch
from monai.networks.nets import SwinUNETR

# libraries to call another python scritp 
# import subprocess
# import sys

# dataloader library to be debugged
from utils.data_utils import get_loader
import argparse

# libarary hosting the brats multi channel transoformation
from monai.transforms.utility.array import ConvertToMultiChannelBasedOnBratsClasses

# to simulate the distributed multi-gpu envionment
# import torch.distributed as dist



def _test_save_image_png():
    # to test save_image_png
    path_2_img = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00085-000/BraTS-GLI-00085-000-seg.nii.gz"
    outname = './qa_output/'+os.path.basename(path_2_img).split(".")[0]+".png"
    # print(outname)
    image=load_1_nifti(path_2_img)
    save_image_png(image, 100, outname)


def save_image_png(img:np.array,slice_number:int,outname:str):
    r'''given a 3D numpy array, it will save a slice as a png image'''
    # Access the image data and display it
    slice = img[:, :,img.shape[2] // 2]
    slice = img[:, :,slice_number]
    matplotlib.image.imsave(outname, slice)
    # matplotlib.image.imsave('./qa_output/'+os.path.basename(path_2_img).split(".")[0]+".png", slice)

def load_1_nifti(path_2_img) -> np.array:
    r"""Load the image using nibabel and return it as a numpy array"""
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
        img_size=96,
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


def _test_compare_pred_and_groundTruth():
    path_pred = "/home/guest183/research-contributions/SwinUNETR/BRATS21/outputs/4gpu_120_epoch/BraTS-GLI-00085-000-seg.nii.gz"
    path_groundTruth = '/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00085-000/BraTS-GLI-00085-000-seg.nii.gz'
    out_dir = "/home/guest183/research-contributions/SwinUNETR/BRATS21/qa_output/"
    compare_pred_and_groundTruth(path_pred, path_groundTruth, out_dir)


def compare_pred_and_groundTruth(path_pred:str, path_groundTruth:str, out_dir:str, slice_number:int):
    r""" this function takes a screen shot of the predicted segmentation from the model and the ground truth segmentation
        as well as the difference between the two. it requires that the basename of the prediction matches the basename of
        the ground truth data. The screenshots are saved in the out_dir path under the folder with patient name.  

    inputs:
        path_pred := path to the output of the swin UNETR model generated by the test.py script
        path_grouondTruth := path to the ground truth segmentation of one BraTS data 

    outputs:
        3 png files saved at out_dir/patient_number

    """
    assert os.path.basename(path_pred) == os.path.basename(path_groundTruth)

    patient_number = "BraTS-GLI-" + "-".join(os.path.basename(path_pred).split("-")[2:4])

    # load prediction, ground truth and subtract them
    pred = load_1_nifti(path_pred)
    groundTruth=load_1_nifti(path_groundTruth)
    difference = pred-groundTruth

    # make patient folder in the out dir
    if not os.path.exists(out_dir+patient_number):
        os.mkdir(out_dir+patient_number)

    # save all the matricies
    save_image_png(pred, slice_number, out_dir+patient_number +"/" + patient_number +"-seg-prediction.png")
    save_image_png(groundTruth, slice_number, out_dir+patient_number +"/" + patient_number + "-seg-groundTruth.png")
    save_image_png(difference, slice_number, out_dir+patient_number +"/" + patient_number + "-seg-difference.png")

def qa_all_predictions(dir_pred_allPatients:str, dir_groundTruth_allPatients, out_dir, slice_number:int):
    r""" given a directory full of predictions, this function will find the ground truth file for each prediction and compares
        the prediction with ground truth over all patients. 
    inputs:
        - dir_pred_allPatients := path to the mother dicrectory containing all the predictions
        - dir_groundTruth_allPatients := path to the directory of all the ground truth data
        - out_dir := path to the output directory where the result of compare_pred_and_groundTruth() is stored. 
        - slice_number := the number of slice along z which will be used for visualization
    """

    pred_all_paths = glob(dir_pred_allPatients+'*.nii.gz')

    for path_pred in pred_all_paths:
        patient_number = "BraTS-GLI-" + "-".join(os.path.basename(path_pred).split("-")[2:4])
        path_groundTruth = dir_groundTruth_allPatients + patient_number + "/" + os.path.basename(path_pred)
        # DEBUGGING{
        # print(path_pred)
        # print(path_groundTruth)
        # break
        # }
        compare_pred_and_groundTruth(path_pred, path_groundTruth, out_dir, slice_number)


def intersection_ratio(pred:torch.Tensor, ground_truth:torch.Tensor) -> float:
    r''' take the inserection between two boolean tensors and 
    devide by the number of True enteries in the ground truth

    '''
    intersection = pred * ground_truth
    ratio = np.sum(intersection==True) / np.sum(ground_truth==True)
    return ratio

def convert_label_to_multiChannel():
   

    return 0

def test_get_loader():
    r''' this function tests the get_loader function in Brats21 SwinUnetr/utils/data_utils.py/get_loader(). 
    inputs:
        - data_dir:str := path to the input data directory
        - datalist_json :json := path to the json dictionary with the path of each input data in each training fold
        -   
    
    '''
    path_data = '/home/odcus/Data/BraTS_Africa_data/'
    path_json = './jsons/brats23_africa_folds.json'
    # test_args = {'data_dir': path_data, 'json_list': path_json,
    #             'fold': 0, 'roi_x': 96, 'roi_y': 96, 'roi_z': 96,
    #             'test_mode': False, 'distributed': True, 'workers': 8,
    #             'batch_size': 1, 
    #              } 
    test_args = argparse.Namespace()
    test_args.data_dir = path_data
    test_args.json_list = path_json
    test_args.fold = 0
    test_args.roi_x = 96
    test_args.roi_y = 96
    test_args.roi_z = 96
    test_args.test_mode = False
    test_args.distributed = False
    test_args.workers = 8
    test_args.batch_size = 1

    # call the get_loader on the arguments
    loader = get_loader(test_args)
    train_loader = loader[0]
    val_loader = loader[1]

    test_results_tensor = torch.zeros([train_loader.sampler.num_samples, 12])

    for idx, batch_data in enumerate(train_loader):
        # print(idx, batch_data.data.numpy().flatten().tolist())        # if isinstance(batch_data, list):

        # extract the image data and the label data from batch
        data, target = batch_data["image"], batch_data["label"]
        # extract the names of the image and label from the batch
        image_dir = batch_data['image_meta_dict']['filename_or_obj']
        label_dir = batch_data['label_meta_dict']['filename_or_obj'][0]
        # seperate each channel
        tumor_core = batch_data["label"][0][0]
        whole_tumor = batch_data["label"][0][1]
        enhancing_tumor = batch_data["label"][0][2]
        # load the raw labels from file:
        label_file_tensor = torch.Tensor(load_1_nifti(label_dir))
        # seperate the raw channels
        # print("a breaking point was here")
        
        # # check if there is non zero elements on any of the labels
        # print(f"number of non-zero voxels in tumor core from get_loader() is: \n {torch.count_nonzero(tumor_core).item()}")
        # print(f"number of non-zero voxels in whole tumor from get_loader() is: \n {torch.count_nonzero(whole_tumor).item()}")
        # print(f"number of non-zero voxels in enhancing tumor from get_loader() is: \n {torch.count_nonzero(enhancing_tumor).item()}")

        test_results_tensor[idx] = torch.Tensor([
            # num voxels in 
                # tumor core
            torch.count_nonzero(tumor_core).item(),
            torch.count_nonzero(label_file_tensor).item(),
                # whole tumor
            torch.count_nonzero(whole_tumor).item(),
            torch.count_nonzero(label_file_tensor).item(),
                # enhancing tumor
            torch.count_nonzero(enhancing_tumor).item(),
            torch.count_nonzero(label_file_tensor).item(),
            0,0,0,0,0,0

            # intersection ratio
            # intersection_ratio()

        ])
    columns = pd.MultiIndex.from_product([['num voxels', 'intersection ratio'], ['TC', 'WT', 'ET'], ['transformed', 'raw']], names=['test', 'tumor subregion', 'source'])
    
    test_results_pd = pd.DataFrame(data=test_results_tensor, columns=columns)
    
    return 0



def main():
    # _test_save_image_png()      # test passed
    # _test_load_patient_to_tensor()   
    # _test_compare_pred_and_groundTruth()

    path_brats="/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
    path_predictions="/home/guest183/research-contributions/SwinUNETR/BRATS21/outputs/4gpu_120_epoch/"
    out_dir = "/home/guest183/research-contributions/SwinUNETR/BRATS21/qa_output/"
    # path_model='/home/guest183/research-contributions/SwinUNETR/BRATS21/runs/4_gpu_4_epochs/model_final.pt'

    # compare all the predictions against ground truth at depth 100. 
    # qa_all_predictions(path_predictions, path_brats, out_dir, 100)

    # match_prediction_name(path_predictions)

    test_get_loader()


 # DO NOT DELETE
if __name__ == "__main__":

    main()

   
    
  