#!/bin/bash

echo "Starting student efficientnet finetuning + head training"
# python main.py --run distill --student efficientnet --seed 42 --mask_ratio 0.75 --dataset idrid
# python main.py --run distill --student efficientnet --seed 42 --mask_ratio 0.75 --dataset aptos
# python main.py --run distill --student efficientnet --seed 42 --mask_ratio 0.75 --dataset mbrset
# python main.py --run distill --student efficientnet --seed 42 --mask_ratio 0.75 --dataset messidor

# python main.py --run head --student efficientnet --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/ktiqi4of/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/ktiqi4of/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/ktiqi4of/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/ktiqi4of/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/ktiqi4of/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize

# python main.py --run head --student efficientnet --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/nuwlywq2/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/nuwlywq2/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/nuwlywq2/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/nuwlywq2/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/nuwlywq2/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize

# python main.py --run head --student efficientnet --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/7wk7l5ue/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/7wk7l5ue/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student efficientnet --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/7wk7l5ue/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student efficientnet --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/7wk7l5ue/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student efficientnet --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/7wk7l5ue/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize

python main.py --run head --student efficientnet --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/d5bbdnd9/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student efficientnet --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/d5bbdnd9/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student efficientnet --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/d5bbdnd9/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student efficientnet --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/d5bbdnd9/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student efficientnet --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/d5bbdnd9/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize


echo "All models finished training!"