#!/bin/bash

echo "Starting student mobilenet finetuning + head training"
python main.py --run distill --student mobilenet --seed 42 --mask_ratio 0.75 --dataset idrid
python main.py --run distill --student mobilenet --seed 42 --mask_ratio 0.75 --dataset aptos
python main.py --run distill --student mobilenet --seed 42 --mask_ratio 0.75 --dataset mbrset
python main.py --run distill --student mobilenet --seed 42 --mask_ratio 0.75 --dataset messidor

python main.py --run head --student mobilenet --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/0hk8494v/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/0hk8494v/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/0hk8494v/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/0hk8494v/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/0hk8494v/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize

python main.py --run head --student mobilenet --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/2axv2zej/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/2axv2zej/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/2axv2zej/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/2axv2zej/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/2axv2zej/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize

python main.py --run head --student mobilenet --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/1z40f1kw/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/1z40f1kw/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/1z40f1kw/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/1z40f1kw/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/1z40f1kw/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize

python main.py --run head --student mobilenet --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/wmn3zkcq/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/wmn3zkcq/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/wmn3zkcq/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/wmn3zkcq/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student mobilenet --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/wmn3zkcq/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize


echo "All models finished training!"