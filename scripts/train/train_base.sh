python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_fp16_4gpu \
    trainer.devices="[4,5,6,7]"
