import tensorflow_hub as hub
import tensorflow_text as tf_tx
import tensorflow as tf
from config import bert_model_name, bert_encoder_layer, bert_preprocess_layer
import os
from get_data import get_data
# to create AdamW optimizer
from official.nlp import optimization

os.environ['PATH'] += os.pathsep + r'D:\servers\graphviz\bin'
tf.get_logger().setLevel('ERROR')


def build_bert_classifier_model(model_name=bert_model_name):
    text_inputs = tf.keras.Input(shape=(), dtype=tf.string, name='input')
    # 注意：默认inputs会被截断到128的长度 此处的outputs是一个字典 keys为['input_type_ids', 'input_word_ids', 'input_mask']
    # key对应的value值都是一个EageTensor shape都是(batch_size,128) 当输入为当句话时,input_type_ids全为0
    outputs = hub.KerasLayer(bert_preprocess_layer[model_name], name='preprocessing')(text_inputs)
    # 注意:bert模型的输出有pooled_output、sequence_output
    # shape of pooled_output = (batch_size,512)
    # shape of sequence_output = (batch_size,128,512)
    outputs = hub.KerasLayer(bert_encoder_layer[model_name], trainable=True, name='bert_encoder')(outputs)
    # 增加一个全连接层
    outputs = tf.keras.layers.Dense(100, activation='relu', name='dense')(outputs['pooled_output'])
    # 电影评论是一个二分类的数据 样本中的label有两种分别为pos和neg 使用0表示neg 1表示pos
    # 最后dense层输出一个值即可 可以看做是回归问题 预测1和0
    outputs = tf.keras.layers.Dense(1, activation=None, name='final')(outputs)
    return tf.keras.Model(text_inputs, outputs)


def train(epochs, init_lr=3e-5):
    class_names, train_ds, val_ds, test_ds = get_data()
    train_ds = train_ds.take(50)
    bert_classifier_model = build_bert_classifier_model()

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    bert_classifier_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw'),
        # 二分类准确率 标签为0和1 阈值默认为0.5
        metrics=tf.metrics.BinaryAccuracy()
    )
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint/',
        monitor='loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq=10
    )
    history = bert_classifier_model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, callbacks=callback)
    return history


def evalute_model(model, test_ds):
    loss, acc = model.evalute(test_ds)
    print(f'Loss: {loss}')
    print(f'Accuracy: {acc}')
    return loss, acc


def plot_model(model):
    tf.keras.utils.plot_model(model)


if __name__ == '__main__':
    model = train(10)
    model.summary()
