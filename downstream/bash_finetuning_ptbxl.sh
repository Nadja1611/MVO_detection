#!/bin/bash

#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-1
#SBATCH --nodelist=mp-gpu4-a6000-1
#SBATCH --job-name=block


#python finetuning_ptbxl.py --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ibk_code_ptbxl_chapman_block/epoch100.pth --dataset ptbxl --data_dir /scratch/nadja/PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ --task multilabel
python pretrain_resnet_ptbxl.py --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ibk_code_ptbxl_chapman_block/epoch100.pth --dataset ptbxl --data_dir /scratch/nadja/PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ --task multilabel
