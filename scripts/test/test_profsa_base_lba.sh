logdir='/log/train/profsa/profsa_release/checkpoints/last.ckpt'

logdir=${logdir//=/\\=}


python train.py \
    experiment=lba30 \
    scheduler.num_warmup_steps=200 \
    optim.lr=0.0002 \
    model.cfg.dropout=0.5 \
    model.cfg.pretrained_weights=$logdir \
    logging.wandb.name=profsa_base_lba30 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba60 \
    scheduler.num_warmup_steps=200 \
    optim.lr=0.0002 \
    model.cfg.dropout=0 \
    model.cfg.pretrained_weights=$logdir \
    logging.wandb.name=profsa_base_lba60 \
    logging.wandb.tags="[lba, lba60]"
