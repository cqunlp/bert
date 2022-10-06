from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from nltk.tokenize import sent_tokenize

pppl_list = []
# cache path
cache_dir="/data0/cache4transformers"
# wiki-dataset
wikidatasets = load_dataset("wikipedia", "20220301.en", split="train", beam_runner='DirectRunner', cache_dir=cache_dir)
# load pretrained model
model = BertForMaskedLM.from_pretrained('google/bert_uncased_L-4_H-256_A-4',cache_dir=cache_dir)
model.eval()
# load pretrained model tokenizer
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4',cache_dir=cache_dir)

def _get_ppl_avg(list1):
    sum = 0
    for i in list1:
        sum += i
    return sum / len(list1)

def sentence_token_nltk(text):
    sent_tokenize_list = sent_tokenize(text)
    return sent_tokenize_list

sentence_loss = 0.
page_pppl = 0.
total_pppl = 0.
total = len(wikidatasets)
print("total pages is {}".format(total))

over512dim_sentence_nums = 0
flag_break = False
one_word_sentence_nums = 0

with torch.no_grad():
    # page loop
    for data in wikidatasets['text']:
        sen_list = sentence_token_nltk(data)
        # sentence loop
        for sentence in sen_list:
            sentence = sentence.replace('\n','')
            nums_sentences = len(sen_list)
            tokenize_input = tokenizer.tokenize(sentence)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            sen_len = len(tokenize_input)
            print(sentence)
            # word loop
            for i, word in enumerate(tokenize_input):
            # add mask to i-th character of the sentence
                tokenize_input[i] = '[MASK]'
                mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
                # print(mask_input.size())
                if mask_input.size()[-1] <= 2:
                    one_word_sentence_nums += 1
                    flag_break = True
                    break
                flag_break = False

                if mask_input.size()[-1] > 512:
                    over512dim_sentence_nums += 1
                    flag_break = True
                    break
                flag_break = False
                output = model(mask_input)

                prediction_scores = output[0]
                softmax = nn.Softmax(dim=0)
                ps = softmax(prediction_scores[0, i]).log()
                word_loss = ps[tensor_input[0, i]]
                sentence_loss += word_loss.item()

                tokenize_input[i] = word
            if flag_break:
                print('The sentence over 512 dim or only 1 word')
                pass
            else:
                ppl = np.exp(-sentence_loss/sen_len)
                # get one sentence LM pppl
                print('The ppl for LM is {}'.format(ppl))
                # initialization
                sentence_loss = 0.
                
                page_pppl += ppl
        if nums_sentences == 0:
                break
        actual_nums = nums_sentences-over512dim_sentence_nums-one_word_sentence_nums
        if actual_nums == 0:
                break
        page_pppl = page_pppl / actual_nums
        print("------------------page total ppl is {}".format(page_pppl))
        pppl_list.append(page_pppl)
        # initialization
        one_word_sentence_nums = 0
        over512dim_sentence_nums = 0
        flag_break = False
        total_pppl += page_pppl
        print(total_pppl)
        page_pppl = 0.

    print('********************The total average ppl for LM is {}'.format(_get_ppl_avg(pppl_list)))
