import re
import torch
import pickle  # 用于保存和加载数据

# 清洗数据：去掉无效字符、注释行、日期等非内容信息。
def token_parser(tokens):
    '''
    input tokens
    清除异常情况（不满足 word/tag 的报错）
    return tokens
    '''
    parsed_tokens = []

    for token in tokens:
        # 使用正则表达式匹配 word/tag 格式
        match = re.match(r"^(.*?)/([^/]+)$", token)
        if match:
            word, tag = match.groups()
            parsed_tokens.append((word, tag))
        else:
            # 如果无法匹配标准格式，尝试处理异常情况
            if "/" in token:
                # 处理多斜杠的情况，取最后一个 / 作为分隔符
                parts = token.rsplit("/", 1)  # 从右向左分割一次
                if len(parts) == 2:
                    word, tag = parts
                    parsed_tokens.append((word, tag))
                else:
                    # 如果仍然无法解析，记录为异常
                    parsed_tokens.append((token, "UNKNOWN_TAG"))
            else:
                # 如果完全缺少 /，记录为异常
                parsed_tokens.append((token, "NO_TAG"))

    return parsed_tokens

# 保留每个词语及其对应的POS标签，将每个词语拆分为字符，并为每个字符分配一个标签（Begin、Middle、End、Single）。
def preprocess_corpus(file_path, is_test):
    '''
    input file_path
    预处理语料库，返回句子和标签
    sentence为去除非中文字符的语句，label为4tag
    return sentences, labels
    '''
    sentences = []
    labels = []
    line_count = 0 # 测试使用

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 读取部分运行，否则运行时间过长
            if is_test:
                line_count += 1
                if line_count > 50:
                    break

            # 去掉空行和注释行
            if not line.strip():
                continue

            line = " ".join(line.split()[1:])
            line = re.sub(r'^\d+-\d+-\d+.*?／w\s+', '', line)

            # 提取句子和分词标签
            tokens = line.strip().split()
            sentence, label = [], []

            tokens = token_parser(tokens)
            for token in tokens:
                (word, tag) = token

                # 空白错误
                if len(word) == 0:
                    continue

                # 对非中文字符预分词
                if re.match(r'[a-zA-Z0-9]+', word) or re.match(r'[^\u4e00-\u9fa5]', word):  # 英文或数字或非中文
                    sentence.append(word)
                    label.append('S')  # 单独成词
                else:
                    # 单字成词：标记为 S
                    if len(word) == 1:
                        sentence.append(word)
                        label.append('S')  # 单字成词
                    # 多字成词：首字标记为 B，中间字标记为 M，尾字标记为 E
                    else:
                        sentence.append(word[0])
                        label.append('B')  # 开始
                        for char in word[1:-1]:
                            sentence.append(char)
                            label.append('M')  # 中间
                        sentence.append(word[-1])
                        label.append('E')  # 结束

            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

def prepro(file_path):
    with open(file_path, 'r', encoding='ansi') as f:
        sentences, labels = [], []
        for line in f:
            # 提取句子
            words = line.strip().split()
            sentence, label = [], []

            for word in words:
                # 对非中文字符预分词
                if re.match(r'[a-zA-Z0-9]+', word) or re.match(r'[^\u4e00-\u9fa5]', word):  # 英文或数字或非中文
                    sentence.append(word)
                    label.append('S')  # 单独成词
                else:
                    # 单字成词：标记为 S
                    if len(word) == 1:
                        sentence.append(word)
                        label.append('S')  # 单字成词
                    # 多字成词：首字标记为 B，中间字标记为 M，尾字标记为 E
                    else:
                        sentence.append(word[0])
                        label.append('B')  # 开始
                        for char in word[1:-1]:
                            sentence.append(char)
                            label.append('M')  # 中间
                        sentence.append(word[-1])
                        label.append('E')  # 结束
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

def save_preprocessed_data(sentences, labels, file_path):
    """保存预处理数据到文件"""
    with open(file_path, 'wb') as f:
        pickle.dump({
            'sentences': sentences,
            'labels': labels
        }, f)

def load_preprocessed_data(file_path):
    """从文件加载预处理数据"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['sentences'], data['labels']

def call_process(file_path, is_test, save_data):
    '''
    input file_path
    对文件进行处理获得编码后标签与句子
    返回编码后句子，编码后标签，词汇表（本文件内）
    ！！注意：编码后句子和标签会有前缀0/-1无实际意义，需模型避免处理
    return encoded_sentences, encoded_labels, vocab
    '''

    # 下次使用时直接加载
    # sentences, labels = load_preprocessed_data('199801_pre.pkl')

    # 采用新格式的数据集
    sentences, labels = prepro(file_path)


    # sentences, labels = preprocess_corpus(file_path, is_test)
    # save_preprocessed_data(sentences, labels, "199801_pre.pkl")

    if save_data:
        # 保存处理后的数据到文件
        print("Saving preprocessed data to files...")
        with open('sentences.pkl', 'wb') as f:
            pickle.dump(sentences, f)
        with open('labels.pkl', 'wb') as f:
            pickle.dump(labels, f)

    return sentences, labels