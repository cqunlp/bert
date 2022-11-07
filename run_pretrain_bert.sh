#Please find all pretrain result in ./outputs/*
time=$(date "+%Y_%m_%d_%H:%M:%S")
if [ ! -d "./outputs" ]; then
       mkdir outputs
fi
pretrain_file_name="pretrain_log"
nohup mpirun -n 8 \
                  python run_pretrain.py \
                         --jit True \
                         --do_train True \
                         --lr 1e-5 \
                         --warmup_steps 10000 \
                         --train_batch_size 256 \
                         --epochs 10 \
                         --save_steps 10000 \
                         --do_load_ckpt False \
                         --model_path {$your_ckpt_path_if_you_need} \
                         --config  {$your_config_json_file_path} \
                         >> ./outputs/${pretrain_file_name}_${time}.log 2>&1 &