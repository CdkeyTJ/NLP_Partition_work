from collections import Counter
import torch
import torch.nn as nn # 导入nn.Module
from sklearn.model_selection import train_test_split # 划分数据集为 训练集 测试集
from torch.utils.data import DataLoader, TensorDataset # 数据加载
import data_preprocess, LSTM_CRF # local file

# 将所有字符映射到唯一的整数ID（词汇表），将标签（B、M、E、S）映射到唯一的整数ID（标签表）
def build_frequency_vocab(sentences):
    '''
    input sentence
    根据句子中字符出现频率构建词汇表和标签映射
    返回vocabulary，字符频率整数表
    return vocab
    '''

    # 统计每个字符出现频率
    counter = Counter([char for sentence in sentences for char in sentence])

    # 根据频率排序，为字符分配整数索引
    vocab = {char: idx + 1 for idx, (char, _) in enumerate(counter.most_common())}
    vocab['<PAD>'] = 0  # 填充字符
    return vocab

def sliding_window(sentence, labels, window_size, stride):
    """
    使用滑动窗口分割句子和标签序列，保留前后文信息。
    """
    sub_sentences, sub_labels = [], []

    # 遍历句子，按滑动窗口分割
    for i in range(0, len(sentence), stride):
        # 获取当前窗口的子句
        sub_sentence = sentence[max(0, i):min(len(sentence), i + window_size)]
        sub_label = labels[max(0, i):min(len(labels), i + window_size)]

        # 填充至窗口大小
        sub_sentence = ['<PAD>'] * max(0, window_size - len(sub_sentence)) + sub_sentence
        sub_label = [-1] * max(0, window_size - len(sub_label)) + sub_label

        sub_sentences.append(sub_sentence)
        sub_labels.append(sub_label)

    return sub_sentences, sub_labels

def encode_data_old(sentences, labels, vocab, label_map, window_size=100, stride=50):
    '''
    使用滑动窗口将句子和标签编码为数字，并保留上下文信息。
    将句子和标签编码为数字
    Return: padded_sentences, padded_labels
    '''
    encoded_sentences, encoded_labels = [], []

    # 遍历每个句子和对应的标签序列
    for sentence, label_seq in zip(sentences, labels):
        # 使用滑动窗口分割句子和标签
        sub_sentences, sub_labels = sliding_window(sentence, label_seq, window_size, stride)

        # 通过词汇表将句子中的字符转为整数，即句子转为整数列表
        for sub_sentence, sub_label in zip(sub_sentences, sub_labels):
            encoded_sentences.append([vocab.get(char, 0) for char in sub_sentence]) #词频编码
            # encoded_sentences.append([0 if char == "<PAD>" else ord(char) for char in sub_sentence]) # ACSII编码
            encoded_labels.append([label_map.get(label, -1) for label in sub_label])

    return encoded_sentences, encoded_labels

def is_valid_split_point(label_seq, i):
    """
    判断是否可以在标签序列的第 i 个位置切分。
    合法切分点包括：S/S, E/B, E/S, S/B。
    """
    if i == 0 or i >= len(label_seq):  # 句子开头或结尾不允许切分
        return False
    prev_label = label_seq[i - 1]
    curr_label = label_seq[i]
    valid_splits = [('S', 'S'), ('E', 'B'), ('E', 'S'), ('S', 'B')]
    return (prev_label, curr_label) in valid_splits


def split_sentence(sentence, labels, max_len):
    """
    根据标签序列和最大长度切分句子，确保不破坏词语完整性。
    """
    sub_sentences, sub_labels = [], []

    start = 0
    while start < len(sentence):
        end = min(start + max_len, len(sentence))  # 默认切分点

        # 如果当前窗口超出了最大长度，寻找最近的合法切分点
        if end < len(sentence) and not is_valid_split_point(labels, end):
            for j in range(end, start, -1):
                if is_valid_split_point(labels, j):
                    end = j
                    break

        # 提取当前窗口的子句和标签
        sub_sentence = sentence[start:end]
        sub_label = labels[start:end]

        # 添加到结果列表
        sub_sentences.append(sub_sentence)
        sub_labels.append(sub_label)

        # 更新起点
        start = end

    return sub_sentences, sub_labels


def encode_data(sentences, labels, vocab, label_map, max_len=100):
    """
    使用合法切分点将句子和标签编码为数字，并保留上下文信息。
    """
    encoded_sentences, encoded_labels = [], []

    # 遍历每个句子和对应的标签序列
    for sentence, label_seq in zip(sentences, labels):
        # 按合法切分点分割句子和标签
        sub_sentences, sub_labels = split_sentence(sentence, label_seq, max_len)

        # 对每个子句进行编码
        for sub_sentence, sub_label in zip(sub_sentences, sub_labels):
            # 填充至最大长度
            sub_sentence = sub_sentence + ['<PAD>'] * (max_len - len(sub_sentence))
            sub_label = sub_label + [-1] * (max_len - len(sub_label))

            # 编码为整数
            encoded_sentences.append([vocab.get(char, 0) for char in sub_sentence])
            encoded_labels.append([label_map.get(label, -1) for label in sub_label])

    return encoded_sentences, encoded_labels

def print_segmentation_results(test_sentences, valid_pred, test_len):
    for i, (sentence, tags) in enumerate(zip(test_sentences, valid_pred)):
        test_len-=1
        if test_len<0:
            break

        print(f"Sentence: {''.join(sentence)}")
        print("result: ", end='')
        # 输出分词结果
        for char, tag in zip(sentence, tags):
            if tag == 3 or tag == 2: # S / E
                print(char, end='/')
            else:
                print(char, end='')
        print()
        print(f"Labelling: {tags}")
        print("-" * 50)

def main():
    # 数据预处理，封装在另一个文件
    # ！！注意：这里的数据只有10条，用于精简预处理(is_test = True时)
    sentences, labels = (data_preprocess.call_process(
            file_path='./icwb2-data/training/pku_training.txt',
            is_test = False,
            save_data = False
        )
    )

    # sentences = [['迈', '向', '充', '满', '希', '望', '的', '新', '世', '纪', '——', '一', '九', '九', '八', '年', '新', '年', '讲', '话', '（', '附', '图', '片', '１', '张', '）'], ['中', '共', '中', '央', '总', '书', '记', '、', '国', '家', '主', '席', '江', '泽', '民']]
    # labels = [['B', 'E', 'B', 'E', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'M', 'M', 'M', 'E', 'B', 'E', 'B', 'E', 'S', 'S', 'B', 'E', 'S', 'S', 'S'], ['B', 'M', 'M', 'E', 'B', 'M', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'B', 'E']]

    # 建立字符整数映射
    vocab = build_frequency_vocab(sentences)
    label_map = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

    # 划分数据集
    train_sentences, test_sentences_ori, train_labels, test_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    # ！！注意：编码后句子和标签会有0/-1无实际意义，需模型避免处理
    train_sentences, train_labels = torch.tensor(encode_data(train_sentences, train_labels, vocab, label_map))
    test_sentences, test_labels = torch.tensor(encode_data(test_sentences_ori, test_labels, vocab, label_map))

    # hyper parameters
    vocab_size = len(vocab)  # 词汇表大小（词频表）
    num_tags = len(label_map)  # 标签类别数
    embedding_dim = 50  # 嵌入维度
    lstm_units = 64  # LSTM 单元数
    max_len = train_sentences.shape[1]  # 最大句子长度
    epochs = 10 # 代数

    # 创建数据加载器，训练时打乱顺序
    batch_size = 32 # 每次加载的数据量
    train_dataset = TensorDataset(train_sentences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_sentences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 模型训练
    model = LSTM_CRF.LSTM_CRF(
        input_size = vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        output_size = num_tags,
        max_len = max_len
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略填充部分 (-1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # 更好的优化器
    from ranger21 import Ranger
    optimizer = Ranger(model.parameters(), lr=0.05)

    model.train_model(train_loader, criterion, optimizer, epochs=epochs)
    pre, rec, f1, valid_pred = model.evaluate_model(test_loader, {0: "B", 1: "M", 2: "E", 3: "S"})
    # word_pos = model.evaluate_model_sentence_convert(vocab, {0: "B", 1: "M", 2: "E", 3: "S"}, test_loader)

    print_segmentation_results(test_sentences_ori, valid_pred, test_len=5)

    # 保存模型的 state_dict 到文件
    # torch.save(model.state_dict(), './model.pth')

if __name__ == '__main__':
    main()