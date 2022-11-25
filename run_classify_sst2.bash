# finetuningSST2/checkpoint文件夹下放预训练好了的bert checkpoint
# finetuningSST2/config文件夹下放对应规格的bert的config.json
time=$(date "+%Y_%m_%d_%H:%M:%S")
cd finetuningSST2
nohup python run_classify.py \
    --bert_ckpt checkpoint/bert_ckpt_step_45w+37w+170000_card_id_2.ckpt \
    --dataset_path SST-2 \
    --config config/bert_config_small.json \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --acc 88 >> finetuningsst2_${time}.log 2>&1 &