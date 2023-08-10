from glob import glob
import json
import random
import  nibabel as nib
import numpy as np


def view_data_shape(path_2_img:str):
    r''' Given path to a nifti file, this function will load the contents as numpy array and prints its shape. 
    '''
    img:np.array = nib.load(path_2_img).get_fdata()
    print(img.shape)
    
    return 0

def kfold_data_dict(data_dir:str, num_folds:int, data_use:str, out_json_file:str=None):
    r''' Given a directory of images from BraTS challege, it will map the patient files and stores them in a dictionary
    inputs:
        - data_dir: path to the training data
        - num_folds: number of k fold cross validation 

    output: 
        k_fold_dict: a dictionary holding the following keys:
            fold
            image
            label
            training
    '''
    patient_dir:list = glob(data_dir + "*/")
    random.shuffle(patient_dir)
    # print(patient_dir)

    # Initialize the dictionary holding k fold cross validation
    kfold_dict:dict = {data_use:[]}
    # figure out how many patients are in a fold
    num_patient_per_fold:int = int(len(patient_dir) / num_folds)
    # print(num_patient_per_fold)

    for k in range(num_folds):
        # print(type(k))
        
        for patient in patient_dir[k*num_patient_per_fold: (k+1)*num_patient_per_fold]:
            # print(patient)
            # print()
            temp_dict = {'fold':k}
            temp_dict["image"] = glob(patient+"/*t*")
            if data_use == "training":
                temp_dict["label"] = glob(patient+"/*seg*")[0]
            kfold_dict[data_use].append(temp_dict)
            # print(temp_dict)
            # break
        # break

    if not (out_json_file is None): 
        with open(out_json_file, 'w') as outfile:
            json.dump(kfold_dict, outfile, indent=4)

    return kfold_dict

def main():

    # check the shape of a single image or segmentation file. 
    # for brats 2021 
    # path_2_img = "/scratch/guest183/BraTS_2021_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/TrainingData/BraTS2021_00000/BraTS2021_00000_seg.nii.gz"
    # for brats 2023 africa 
    # path_2_img = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00387-000/BraTS-GLI-00387-000-seg.nii.gz"
    # view_data_shape(path_2_img)

    # generate json file for GLI data containing the path to n fold cross validation data 
    # data_dir_gli_training:str = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
    # kfold_dict_gli_training:dict = kfold_data_dict(data_dir_gli_training, 5, out_json_file="brats23_africa_folds.json")

    # generate json for GLI validation folder
    data_dir_GLI_val:str = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/"
    json_outdir:str = "./jsons/brats23_gli_test.json"
    kfold_dict_GLI_testing:dict = kfold_data_dict(data_dir_GLI_val, 1, "testing", json_outdir)

    # Generate json file for sub saharan africa data

    

# DO NOT DELETE
if __name__ == "__main__":
    main()
