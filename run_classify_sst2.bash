cd finetuningSST2
python run_classify.py \
    --bert_ckpt checkpoint/{$the_pretrain_bert_ckpt_path} \
    --dataset_path SST-2 \
    --config ../config/{$the_bert_config_json_file_path} \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5
