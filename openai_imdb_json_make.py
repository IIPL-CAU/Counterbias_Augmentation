from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from random import shuffle

import json
import jsonlines

def main():
    # Prepare
    original_text = list()
    aug_graph_list = list()
    aug_impt_graph_list = list()
    aug_text_list = list()
    error_count = 0

    # Load the dataset
    ds = load_dataset("stanfordnlp/imdb")

    system_message = """

    You are a chatbot used for data augmentation. Your job is reconstructing the selected triples into a sentence or paragraph.

    """

    graph_content_message = """Please create sentences for the following Triples. 

    Here is an example of a Graph Triplet and its corresponding reconstructed text: 

    1. A few people | are in | a restaurant setting 
    2. One person | is drinking | orange juice  
    Output format: 
    A few people in a restaurant setting, one of them is drinking orange juice. 

    Here is another example of a Graph Triplet and its corresponding reconstructed text: 
    2. I | am | a student 
    1. I | am | a professor 
    Output format: 
    I am a student and also a professor. 

    Please provide no answers other than the reconstructed text. Output only the reconstructed text. And don't consider the number of sentences in the input text. 

    Please follow the order of the inputs strictly as they are written. Do not consider the numbers provided in the inputs. For example: 
    2. I | am | a student 
    1. I | am | a professor 
    Output format: 
    I am a student and also a professor. 
    In this case, even though the sequence numbered ``2" comes first numerically, ignore the numbers and generate the output starting with "I | am | a student" as shown in the example. 

    Here is an graph triplet you should make a text:

    """

    #
    json_dict = dict()
    dict_list = list()

    #
    for i, text in enumerate(tqdm(ds['train']['text'])):
        # max_length truncates the text to 300 tokens
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        inputs = bert_tokenizer(text, return_tensors="pt", max_length=300, truncation=True)
        input_text = bert_tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

        original_text.append(input_text)

        graph_content_message_include_text = graph_content_message + input_text
        imp_graph_content_message_include_text = imp_graph_content_message + input_text

        input_dict = {
            'custom_id': "request-{:n}".format(i),
            'method': 'POST',
            'url': "/v1/chat/completions",
            'body': {'model': 'gpt-4o-mini',
                    'messages': [{'role': 'system', 'content': system_message},
                                {'role': 'user', 'content': graph_content_message_include_text}],
                                "max_tokens": 1000
                    }
        }
        dict_list.append(input_dict)

    with open("imdb_openai.jsonl", "w", encoding="utf-8") as f:
        for d in dict_list:
            f.write(json.dumps(d) + "\n")

    ####################################################################################################################################
    # Here is for the output of the OpenAI



    graph_out = list()
    imp_graph = list()

    with jsonlines.open("imdb_graph_output.jsonl") as f:
        for line in f.iter():
            graph_out.append(line)

    with jsonlines.open("imdb_openai_imp_triplet.jsonl") as f:
        for line in f.iter():
            imp_graph.append(line)

    processed_data = list()
    processed_imp_data = list()
    processed_changed_data = list()
    error_count = 0

    for i in range(len(graph_out)):
        processed_data.append(graph_out[i]['response']['body']['choices'][0]['message']['content'].split('\n\n')[0])

    for i in range(len(imp_graph)):
        processed_imp_data.append(imp_graph[i]['response']['body']['choices'][0]['message']['content'].split('\n\n')[0])

    for i in range(len(imp_graph)):
        try:
            processed_changed_data.append(imp_graph[i]['response']['body']['choices'][0]['message']['content'].split('\n\n')[1])
        except IndexError:
            error_count += 1
            processed_changed_data.append('')

    error_count = 0

    for i in range(len(processed_data)):
        all_graph_split = processed_data[i].split('\n')
        imp_graph_split = processed_imp_data[i].split('\n')
        try:
            change_graph_split = processed_changed_data[i].split('\n')

            change_index = [int(x.split('.')[0])-1 for x in imp_graph_split[1:]]

            for j in range(len(change_index)):
                processed_data[i] = processed_data[i].replace(all_graph_split[change_index[j]], change_graph_split[j+1])
            # j = random.choice(range(len(change_index)))
            # processed_data[i] = processed_data[i].replace(all_graph_split[change_index[j]], change_graph_split[j+1])
        except:
            print(i)

    male_list = list()
    female_list = list()

    with open('male_word_file.txt', 'r') as f:
        male_dat = f.readlines()

    for m in male_dat:
        m = m.replace('\n' , '')
        male_list.append(m)
        

    with open('female_word_file.txt', 'r') as f:
        female_dat = f.readlines()

    for f in female_dat:
        f = f.replace('\n' , '')
        female_list.append(f)

    male_list = male_list + [m.capitalize() for m in male_list]
    female_list = female_list + [m.capitalize() for m in female_list]

    gender_processed_data = list()

    for i in range(len(processed_data)):
        long_string = processed_data[i]

        # Split the string into individual words
        words = long_string.split(' ')

        # Iterate over the words and replace them if they match any word in the male_list or female_list
        for i in range(len(words)):
            if words[i] in male_list:
                words[i] = female_list[male_list.index(words[i])]
            elif words[i] in female_list:
                words[i] = male_list[female_list.index(words[i])]

        # Join the modified words back into a string
        modified_string = ' '.join(words)

        gender_processed_data.append(modified_string)

    shuffled_gender_processed_data = list()

    for p in gender_processed_data:
        p_split = p.split('\n')
        shuffle(p_split)
        shuffled_gender_processed_data.append('\n'.join(p_split))

    error_count = 0

    reindex_shuffled_gender_processed_data = list()

    for t in shuffled_gender_processed_data:
        pre_t = list()
        try:
            for i, tt in enumerate(t.split('\n')):
                pre_t.append(str(i+1) + '.' + tt.split('.')[1])
        except:
            pre_t = t
            error_count += 1

        reindex_shuffled_gender_processed_data.append('\n'.join(pre_t))

    system_message = """

    You are a chatbot used for data augmentation. Your job is reconstructing the selected triples into a sentence or paragraph.

    """

    reconstruct_message = """Please create sentences for the following Triples. 

    Here is an example of a Graph Triplet and its corresponding reconstructed text: 

    1. A few people | are in | a restaurant setting 
    2. One person | is drinking | orange juice  
    Output format: 
    A few people in a restaurant setting, one of them is drinking orange juice. 

    Here is another example of a Graph Triplet and its corresponding reconstructed text: 
    2. I | am | a student 
    1. I | am | a professor 
    Output format: 
    I am a student and also a professor. 

    Please provide no answers other than the reconstructed text. Output only the reconstructed text. And don't consider the number of sentences in the input text. 

    Please follow the order of the inputs strictly as they are written. Do not consider the numbers provided in the inputs. For example: 
    2. I | am | a student 
    1. I | am | a professor 
    Output format: 
    I am a student and also a professor. 
    In this case, even though the sequence numbered ``2" comes first numerically, ignore the numbers and generate the output starting with "I | am | a student" as shown in the example. 

    Here is an graph triplet you should make a text:
        
    """

    dict_list3 = list()

    for i, triplet in enumerate(tqdm(shuffled_gender_processed_data)):

        message_text = reconstruct_message + shuffled_gender_processed_data[i]

        input_dict = {
            'custom_id': "request-{:n}".format(i),
            'method': 'POST',
            'url': "/v1/chat/completions",
            'body': {'model': 'gpt-4o-mini',
                    'messages': [{'role': 'system', 'content': system_message},
                                {'role': 'user', 'content': message_text}],
                                "max_tokens": 2000
                    }
        }
        dict_list3.append(input_dict)

    with open("shuffled_imdb_triplet_openai_input2.jsonl", "w", encoding="utf-8") as f:
        for d in dict_list3:
            f.write(json.dumps(d) + "\n")

if __name__=='__main__':
    main()