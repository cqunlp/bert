# finetuningSST2/checkpoint文件夹下放预训练好了的bert checkpoint 
##当然你也可以随便搞 路径对就行了
# finetuningSST2/config文件夹下放对应规格的bert的config.json
##当然你也可以随便搞 路径对就行了
# 所有的结果在 finetuningSST2/result_log/*
time=$(date "+%Y_%m_%d_%H:%M:%S")
# if [ ! -d "finetuningSST2/checkpoint" ]; then
#        mkdir finetuningSST2/checkpoint
# fi
cd finetuningSST2
nohup python run_classify.py \
    --device_target Ascend \
    --amp True \
    --bert_ckpt {这里写ckpt路径} \
    --dataset_path sst-2 \
    --max_length 64 \
    --config {这里写config路径} \
    --train_batch_size 16 \
    --test_batch_size 16 \
    --epochs 4 \
    --lr 2e-5 \
    --acc {这里写需要达标的精度} >> result_log/finetuningsst2_${time}.log 2>&1 &
