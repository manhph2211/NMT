import os
import numpy as np 


def load_data(src_file, trg_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        list_src = f.readlines()
    
    with open(trg_file, 'r', encoding='utf-8') as f:
        list_trg = f.readlines()
    
    data  =	[list_src,list_trg]

    return data


model_name = './translate_en_vi_converter'
tokenizers = tf.saved_model.load(model_name)


if __name__ == '__main__':

	test_data = load_data("./data/test.en", "data/test.vi")
	en,vi = test_dataset
	for i,(en_sentence,vi_sentence) in enumerate(zip(en,vi)):
		print('----------------------------------------------')
		print("English: ", en_sentence)
		print("Vietnamese: ", vi_sentence)
		print('----------------------------------------------')
		if i==2:
			break
