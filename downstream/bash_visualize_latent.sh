#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-1
#SBATCH --job-name=block

#python visualize_embeddings.py --ckpt_dir /home/nadja/MVO_Project_multilabel/downstream/weights_after_finetuning/fine9.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel/fold_MVO_1 --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
python visualize_embeddings.py --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ptbxl_stemis/ptbxl_onlySTEMI5.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest/fold_MVO_1 --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
