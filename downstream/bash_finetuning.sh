#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a6000
#SBATCH --nodelist=mp-gpu4-a6000-2
#SBATCH --job-name=block
#python train_linear_eval.py --data_percentage 1.0 --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/ptbxl_fine --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ptbxl_stemis/ptbxl_onlySTEMINormal4.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel

#python finetuning_jepa_mvo_multilabel.py --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/ptbxl_fine --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/IBK/epoch400.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_latest --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#python train_jepa_multilabel.py --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/resent --dropout 0.1  --model_name ejepa_random --data_percentage 1.0 --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/IBK/epoch90.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel 
#python train_resnet.py --data_percentage 1.0 --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/resent --dropout 0.25  --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ibk_code_ptbxl_chapman_block/epoch75.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#python linear_eval_mvo.py --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/ptbxl_fine --model_name ejepa_random --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ptbxl_fine/ptbxl6.pth --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel/ --pathology mvo --dataset mvo --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ --task multilabel
#!/bin/bash

for epoch in {1..9..1}; do
  echo "Running for epoch${epoch}.pth"
  python train_jepa_multilabel.py \
    --metrics_dir /home/nadja/MVO_Project_multilabel/metrics/resent \
    --dropout 0.1 \
    --model_name ejepa_random \
    --data_percentage 1.0 \
    --ckpt_dir /home/nadja/MVO_Project_multilabel/weights/ptbxl_stemis/ptbxl_all_${epoch}.pth \
    --data_mvo /home/nadja/ECG_JEPA_Git/downstream/data_all/umbrella_multilabel_earliest \
    --pathology mvo \
    --dataset mvo \
    --data_dir /gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/ \
    --task multilabel
done
