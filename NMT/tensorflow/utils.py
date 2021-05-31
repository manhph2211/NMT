import tensorflow as tf


model_name = 'translate_en_vi_converter'
tokenizers = tf.saved_model.load(model_name)


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


