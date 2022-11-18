# finetuningSST2/checkpoint文件夹下放预训练好了的bert checkpoint
# finetuningSST2/config文件夹下放对应规格的bert的config.json
cd finetuningSST2
python run_classify.py \
    --bert_ckpt checkpoint/bert_L4_H128_step_308749_card_id_0.ckpt \
    --dataset_path SST-2 \
    --config config/bert_L4_H128_config.json \
    --batch_size 16 \
    --epochs 10 \
    --lr 5e-5 \
    --acc 85