# TFRecord使用二进制编码数据,读取之后只占用一块内存块,加载速度快
import json
import sys

import tensorflow as tf
import bert.tokenization as tokenization
import bert.optimization as optimization

# 原始预料路径
CORPUS_FILE = './data/train_data.json'
# 保存的TFRecord文件路径
TFRECORD_FILE = './data/train_data.tfrecord'
# 三元组的Schema文件路径
SCHEMA_FILE = './data/all_50_schemas'
# bert_tokenizer分词器按词汇表进行分词,提供将词转为id列表和id列表转词列表
BERT_TOKENIZER = tokenization.FullTokenizer(vocab_file="./data/chinese_L-12_H-768_A-12/vocab.txt", do_lower_case=True)
# 设置最大的语句长度
SEQUENCE_LENGTH = 100

# 50个schema 可以在调用read_schema之后获取到真实的schema个数 再修改该全局变量
SCHEMA_SIZE = 50


def read_schema(schema_file=SCHEMA_FILE):
    schema2id = {}
    id2schema = {}
    with open(schema_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            line = json.loads(line.strip())
            subject_type = line['subject_type']
            object_type = line['object_type']
            predicate = line['predicate']
            spo = f'{subject_type}-{predicate}-{object_type}'
            schema2id[spo] = index
            id2schema[index] = spo
    return schema2id, id2schema


# 读取原始预料并进行处理
def process_data(schema2id, corpus_file=CORPUS_FILE):
    with open(corpus_file, 'r', encoding='utf-8') as cf:
        lines = cf.readlines()
    # 定义一个列表存储每个样本数据
    # 文本样本->字分词->id列表->统一长度为SEQUENCE_LENGTH
    inputs_padded = []
    # 定义一个列表存储每个列表在统一长度时,用PAD进行填充的位置
    inputs_masked = []
    # 定义一个列表存储每个样本的标签 标签使用one-hot编码
    outputs_spo_class = []
    for line in lines:
        line = json.loads(line.strip())
        # 每个样本的文本数据
        sentence = line['text']
        # 使用bert_tokenizer对sentence进行分词并转为id
        tokens = BERT_TOKENIZER.tokenize(sentence)
        token_ids = BERT_TOKENIZER.convert_tokens_to_ids(tokens)
        # 统计语句分词之后的长度 使用0进行填充 查看vocab.txt可以看到PAD处于第一行 对应索引为0
        token_padding_ids = token_ids[:SEQUENCE_LENGTH] \
            if len(token_ids) >= SEQUENCE_LENGTH \
            else token_ids + [0] * (SEQUENCE_LENGTH - len(token_ids))
        inputs_padded.append(token_padding_ids)
        # 为每个样本样本数据指定mask mask标识那些PAD的位置 之后的处理过程中需要进行忽略
        token_mask_ids = [0] * SEQUENCE_LENGTH \
            if len(token_ids) >= SEQUENCE_LENGTH \
            else [0] * len(token_ids) + [1] * (SEQUENCE_LENGTH - len(token_ids))
        inputs_masked.append(token_mask_ids)
        # 每个样本的标签,是一个列表
        spo_list = line['spo_list']
        # 初始化一个长度为schema个数的列表 记录文本数据对应的schema
        # 一个文本可能对应多个schema 每个schema使用one-hot编码 然后one-hot编码的向量相加
        spo = [0] * len(schema2id)
        # 将spo_list中的schema转为id 然后将spo对应id索引位置的编码设置为1
        spo_ids = [schema2id[f"{spo_item['subject_type']}-{spo_item['predicate']}-{spo_item['object_type']}"] for
                   spo_item in spo_list]
        for spo_id in spo_ids:
            spo[spo_id] = 1
        outputs_spo_class.append(spo)
    return inputs_padded, inputs_masked, outputs_spo_class


if __name__ == '__main__':
    schema2id, id2schema = read_schema()
    print(len(schema2id))  # 50
    print(schema2id)
    print(id2schema)

    inputs_padded, inputs_masked, outputs_spo_class = process_data(schema2id)
    print(inputs_padded[3])
    print(inputs_masked[3])
    print(outputs_spo_class[3])
