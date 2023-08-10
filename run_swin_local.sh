source /home/odcus/Software/Kilimanjaro_swinUNETR/SWIN_ENV/bin/activate
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

path_swin='/home/odcus/Software/Kilimanjaro_swinUNETR/'
path_data='/home/odcus/Data/BraTS_Africa_data/'

# code to train the model
python $path_swin'main.py'\
 --pretrained_dir=$path_swin'/pretrained_models'\
 --pretrained_model_name='model-epoch100-baseModel-2023.pt'\
 --resume_ckpt --distributed --lrschedule='warmup_cosine'\
 --json_list=$path_swin'jsons/brats23_africa_folds.json'\
 --sw_batch_size=8\
 --batch_size=2\
 --data_dir=$path_data\
 --val_every=1\
 --infer_overlap=0.7\
 --out_channels=3 --in_channels=4 --spatial_dims=3\
 --save_checkpoint --use_checkpoint\
 --feature_size=48 --max_epochs=1 --logdir='local_resumeCheckpoint_test'

# options for training
#   to continue the training of a pre-trained model:
# --pretrained_dir='/home/odcus/Software/Kilimanjaro_swinUNETR/pretrained_models' --pretrained_model_name=? --resume_ckpt




# code to generate predictions by the model
# python $path_swin'test.py' --data_dir=$path_data --exp_name='4gpu_120_epoch' --json_list=$path_swin'jsons/brats23_africa_validation_folds.json' --pretrained_dir=$path_swin'runs/4_gpu_120_epochs' --pretrained_model_name='model_final.pt' --infer_overlap=0.7

# code to run quality assurance on the results of the test
# python $path_swin'quality_assurance.py'
