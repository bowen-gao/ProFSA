logdir='/log/train/profsa/profsa_release'

python test.py $logdir \
    --update_func test_kahraman test_toughm1 \
    --ckpt last
