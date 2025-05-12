#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a100
#SBATCH --nodelist=mp-gpu4-a100-1
#SBATCH --job-name=block

#python train_linear_eval.py --data_percentage 1.0 --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/ptbxl_fine --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ptbxl_stemis/ptbxl_onlySTEMINormal4.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel

python train_jepa_multilabel.py --data_percentage 1.0 --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/resent --dropout 0.1  --model_name ejepa_random  --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/PTBXL/epoch70.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel 
#python train_resnet.py --data_percentage 0.25 --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/resent --dropout 0.25  --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/downstream/output/finetuning/resnet_ptbxl_all_75.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#!/bin/bash

