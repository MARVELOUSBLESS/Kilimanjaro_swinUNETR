source /home/odcus/Software/Kilimanjaro_swinUNETR/SWIN_ENV/bin/activate
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

path_swin='/home/odcus/Software/Kilimanjaro_swinUNETR/'
path_data='/home/odcus/Data/BraTS_Africa_data/'

# code to train the model
python $path_swin'main.py' --distributed --lrschedule='cosine_anneal' --json_list=$path_swin'jsons/brats23_africa_folds.json' --sw_batch_size=8 --batch_size=4 --data_dir=$path_data --val_every=2 --infer_overlap=0.7 --out_channels=3 --in_channels=4 --spatial_dims=3 --save_checkpoint --use_checkpoint --feature_size=24 --max_epochs=2 --logdir='local_epoch_10'

# code to generate predictions by the model
# python $path_swin'test.py' --data_dir=$path_data --exp_name='4gpu_120_epoch' --json_list=$path_swin'jsons/brats23_africa_validation_folds.json' --pretrained_dir=$path_swin'runs/4_gpu_120_epochs' --pretrained_model_name='model_final.pt' --infer_overlap=0.7

# code to run quality assurance on the results of the test
# python $path_swin'quality_assurance.py'