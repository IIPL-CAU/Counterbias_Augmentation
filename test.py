import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model3 = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

checkpoint = torch.load('/nas_homes/kyohoon1/counterfactual/model_checkpoint/imdb/bert/cls_training_checkpoint_seed_42_aug_False_ours.pth.tar')
model1.load_state_dict(checkpoint['model'])

checkpoint = torch.load('/nas_homes/kyohoon1/counterfactual/model_checkpoint/imdb/bert/cls_training_checkpoint_seed_42_aug_True_LLM.pth.tar')
model2.load_state_dict(checkpoint['model'])

checkpoint = torch.load('/nas_homes/kyohoon1/counterfactual/model_checkpoint/imdb/bert/cls_training_checkpoint_seed_42_aug_True_ours.pth.tar')
model3.load_state_dict(checkpoint['model'])

model1.to('cuda')
model2.to('cuda')
model3.to('cuda')

# test_dat = pd.read_csv('../counterfactual/human_cad_snli_test.tsv', sep='\t')
# ds = load_dataset('SetFit/sst2')
# test_dat = ds['test']
ds = load_dataset("Yelp/yelp_review_full")
test_dat = ds['test']

correct = 0
wrong = 0

text_list = test_dat['text']#.tolist()
label_list = test_dat['label']#.tolist()

for ii in tqdm(range(len(test_dat)), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
    src_sequence = tokenizer(text_list[ii], padding=True, truncation=True, return_tensors='pt', max_length=300)['input_ids'].to('cuda')

    with torch.no_grad():
        #  convert the model output logits to predicted class labels
        logit = model1(input_ids=src_sequence)['logits']

        predictions = logit.argmax(dim=1)[0]

        if predictions in [0,1]:
            predictions_ = 0
        if predictions in [2,3,4]:
            predictions_ = 1

        if label_list[ii] in [0,1]:
            gt = 0
        if label_list[ii] in [2,3,4]:
            gt = 1

        if predictions_ != gt:
            wrong += 1
        else:
            correct += 1

print(correct / (correct + wrong))

correct = 0
wrong = 0

text_list = test_dat['text']#.tolist()
label_list = test_dat['label']#.tolist()

for ii in tqdm(range(len(test_dat)), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
    src_sequence = tokenizer(text_list[ii], padding=True, truncation=True, return_tensors='pt', max_length=300)['input_ids'].to('cuda')

    with torch.no_grad():
        #  convert the model output logits to predicted class labels
        logit = model2(input_ids=src_sequence)['logits']

        predictions = logit.argmax(dim=1)[0]

        if predictions in [0,1]:
            predictions_ = 0
        if predictions in [2,3,4]:
            predictions_ = 1

        if label_list[ii] in [0,1]:
            gt = 0
        if label_list[ii] in [2,3,4]:
            gt = 1

        if predictions_ != gt:
            wrong += 1
        else:
            correct += 1

print(correct / (correct + wrong))

correct = 0
wrong = 0

text_list = test_dat['text']#.tolist()
label_list = test_dat['label']#.tolist()

for ii in tqdm(range(len(test_dat)), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
    src_sequence = tokenizer(text_list[ii], padding=True, truncation=True, return_tensors='pt', max_length=300)['input_ids'].to('cuda')

    with torch.no_grad():
        #  convert the model output logits to predicted class labels
        logit = model3(input_ids=src_sequence)['logits']

        predictions = logit.argmax(dim=1)[0]

        if predictions in [0,1]:
            predictions_ = 0
        if predictions in [2,3,4]:
            predictions_ = 1

        if label_list[ii] in [0,1]:
            gt = 0
        if label_list[ii] in [2,3,4]:
            gt = 1

        if predictions_ != gt:
            wrong += 1
        else:
            correct += 1

print(correct / (correct + wrong))