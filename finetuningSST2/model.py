from typing import Optional
import mindspore
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from src.bert import BertForPretraining

class BertBinaryClassificationModel(nn.Cell):
    def __init__(self, config, ckpt_file: Optional[str] = None):
        super().__init__()
        self.dropout = nn.Dropout(0.8)
        # load ckpt
        self.config = config
        self.classifier = nn.Dense(self.config.hidden_size, 2, weight_init=TruncatedNormal(config.initializer_range))
        model = BertForPretraining(self.config)
        if ckpt_file is not None:
            bert_dict = mindspore.load_checkpoint(ckpt_file)
            mindspore.load_param_into_net(model, bert_dict)
        self.bert = model.bert

    def construct(self, input_ids, attention_mask=None, token_type_ids=None,
                  position_ids=None, head_mask=None, label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.dropout(logits)
        outputs = (logits,) + outputs[2:]

        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            # print(logits.shape,label.shape,label.view(-1).shape)
            # label.view(-1) == (1024,) 
            # logits == (16, 2)
            # before (16, 2) (16, 1) (16,)
            # after (16, 2) (16, 64) (1024,)
            loss = loss_fct(logits.view(-1, 2), label.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
