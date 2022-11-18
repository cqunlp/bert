import mindspore
import mindspore.dataset as ds

from src.config import PretrainedConfig

if __name__ == "__main__":
    config = PretrainedConfig()
    dataset_path = config.dataset_mindreocrd_dir
    train_dataset = ds.MindDataset(dataset_files=dataset_path)
    output_types = train_dataset.output_types()
    print(output_types)
    train_dataset.save(file_name="merge_and_save_mindrecord/bert_pretrain_data.mindrecord",
                       num_files=8,
                       file_type='mindrecord')
