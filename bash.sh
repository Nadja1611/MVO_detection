#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a100
#SBATCH --nodelist=mp-gpu4-a100-2
#SBATCH --job-name=random


#python pretrain_ECG_JEPA.py --mask_type random --mask_scale 0.6 0.7 --batch_size 128 --lr 1e-5 --data_dir_shao /gpfs/data/fs72515/nadja_g/ECG_JEPA/subset_shao/  --data_dir_code15 /gpfs/data/fs72515/nadja_g/ECG_JEPA/subset_code15/
#python pretrain_ECG_JEPA_500Hz.py --mask_type random --mask_scale 0.6 0.7 --batch_size 128  --data_dir_shao /gpfs/data/fs72515/nadja_g/ECG_JEPA/shaoxing/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/  --data_dir_code15 /gpfs/data/fs72515/nadja_g/ECG_JEPA/code-15/

#python pretrain_ECG_JEPA.py --mask_type random --mask_scale 0.6 0.7 --batch_size 128 --lr 1e-5 --data_dir_shao /gpfs/data/fs72515/nadja_g/ECG_JEPA/subset_shao/  --data_dir_code15 /gpfs/data/fs72515/nadja_g/ECG_JEPA/subset_code15/
#python pretrain_ECG_JEPA_500Hz_ptbxl.py --mask_type block --mask_scale 0.6 0.7 --batch_size 128  --data_dir_mimic /scratch/nadja/MIMICIV/files/ --data_dir_ptbxl /scratch/nadja/PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ --data_dir_shao /scratch/nadja/shao/shao_dataset/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/  --data_dir_code15 /scratch/nadja/code-15/
python pretrain_ECG_JEPA_500Hz_ptbxl_cpsc_ibk.py --mask_type random --mask_scale 0.6 0.7 --batch_size 128   --data_dir_ibk /scratch/nadja/IBK_EKG_Export/all/  --data_dir_cpsc /scratch/nadja/cpsc/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/ --data_dir_ptbxl /scratch/nadja/PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ --data_dir_shao /scratch/nadja/shao/shao_dataset/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/  --data_dir_code15 /scratch/nadja/code-15/
#python pretrain_ECG_JEPA_500Hz_ptbxl_cpsc.py --mask_type block --mask_scale 0.6 0.7 --batch_size 128     --data_dir_cpsc /scratch/nadja/cpsc/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/ --data_dir_ptbxl /scratch/nadja/PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ --data_dir_shao /scratch/nadja/shao/shao_dataset/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/  --data_dir_code15 /scratch/nadja/code-15/
#python pretrain_ECG_JEPA_500Hz.py --mask_type random --mask_scale 0.6 0.7 --batch_size 128  --data_dir_shao /scratch/nadja/shao/shao_dataset/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/  --data_dir_code15 /scratch/nadja/code-15/
