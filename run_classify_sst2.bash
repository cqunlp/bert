# finetuningSST2/checkpoint文件夹下放预训练好了的bert checkpoint
# finetuningSST2/config文件夹下放对应规格的bert的config.json
# 所有的结果在finetuningSST2/result_log
time=$(date "+%Y_%m_%d_%H:%M:%S")
cd finetuningSST2
nohup python run_classify.py \
    --bert_ckpt checkpoint/bert_ckpt_step_170000_card_id_7.ckpt \
    --dataset_path sst-2 \
    --max_length 512 \
    --config config/bert_config_small.json \
    --train_batch_size 16 \
    --test_batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --acc 88 >> result_log/finetuningsst2_${time}.log 2>&1 &