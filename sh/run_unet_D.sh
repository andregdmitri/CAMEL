#!/bin/bash

echo "Starting student unet finetuning + head training"
# python main.py --run distill --student unet --seed 42 --mask_ratio 0.75 --dataset idrid
# python main.py --run distill --student unet --seed 42 --mask_ratio 0.75 --dataset aptos
# python main.py --run distill --student unet --seed 42 --mask_ratio 0.75 --dataset mbrset
# python main.py --run distill --student unet --seed 42 --mask_ratio 0.75 --dataset messidor

# python main.py --run head --student unet --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/r8elpwk3/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/r8elpwk3/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/r8elpwk3/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/r8elpwk3/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/r8elpwk3/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize

# python main.py --run head --student unet --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/q32z8161/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/q32z8161/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/q32z8161/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/q32z8161/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/q32z8161/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize

# python main.py --run head --student unet --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/vd7bha7h/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/vd7bha7h/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/vd7bha7h/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/vd7bha7h/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize
# python main.py --run head --student unet --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --load_backbone ./distillation/vd7bha7h/checkpoints/best_distillation_42_mbrset.ckpt --num_workers 8 --prune --quantize

# python main.py --run head --student unet --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/27qqo040/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student unet --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/27qqo040/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student unet --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/27qqo040/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student unet --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/27qqo040/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student unet --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --load_backbone ./distillation/27qqo040/checkpoints/best_distillation_42_messidor.ckpt --num_workers 8 --prune --quantize


echo "All models finished training!"