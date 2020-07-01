python train.py --gpus 1 --model film --dataset nlvr2 --epochs 30 --run_name NLVR2
python train.py --gpus 1 --model butd --dataset nlvr2 --epochs 30 --run_name NLVR2

python train.py --gpus 1 --model film --dataset vqa2 --epochs 30 --run_name VQA2
python train.py --gpus 1 --model butd --dataset vqa2 --epochs 30 --run_name VQA2

python train.py --gpus 1 --model film --dataset gqa --epochs 15 --run_name GQA
python train.py --gpus 1 --model butd --dataset gqa --epochs 15 --run_name GQA
