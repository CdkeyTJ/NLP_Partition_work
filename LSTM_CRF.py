import time # 记录时间
from tqdm import tqdm # 进度条
import torch
import torch.nn as nn # 导入nn.Module
from collections import Counter # 提高评估指标计算效率
from sklearn.metrics import precision_recall_fscore_support # 增强模型评估速度
from TorchCRF import CRF
import CRF_realize # local file

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM_CRF(nn.Module):
    def __init__(self, input_size, embedding_dim, lstm_units, output_size, max_len):
        """
        初始化 LSTM 模型。

        参数:
        - vocab_size: 词汇表大小 (int)
        - embedding_dim: 嵌入维度 (int)
        - lstm_units: LSTM 单元数 (int)
        - num_tags: 标签类别数 (int)
        - max_len: 输入序列的最大长度 (int)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_len = max_len

        # embedding层：字符映射至向量
        self.embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_dim,
            padding_idx=0 # 指定填充符号 <PAD> 的索引为 0，其嵌入向量始终为零
        )
        # LSTM层：获取上下文信息
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=2
        )
        # 全连接层
        self.fc = nn.Linear(lstm_units, output_size)

        # 更严格的CRF初始化
        # self.crf = CRF(num_labels=output_size)

    def forward(self, x, tags=None, mask=None):
        x = x.to(device)
        # 向量化
        embeds = self.embedding(x)          #
        lstm_out, _ = self.lstm(embeds)     #
        emissions = self.fc(lstm_out)       #
        # logits = self.crf(emissions)
        return emissions


    def decode(self, emissions, mask=None):
        """Use CRF to decode the best path"""
        return self.crf.decode(emissions, mask)

    def train_model(self, train_loader, criterion, optimizer, epochs=5):
        self.to(device)
        self.train()

        start_training_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time() # 计时
            epoch_loss = 0
            for sentences, labels in train_loader:
                # 确保数据移动到目标设备
                sentences, labels = sentences.to(device), labels.to(device)

                optimizer.zero_grad()

                # 将 labels 的形状从 [batch_size, sequence_length] 转换为 [batch_size * sequence_length]
                # labels = labels.view(-1)
                # outputs = self(sentences).view(-1)
                outputs = self.forward(sentences)
                batch_size, sequence_length, num_classes = outputs.shape
                # 展平 outputs 和 labels
                outputs = outputs.view(-1, num_classes)  # [batch_size * sequence_length, num_classes]
                labels = labels.view(-1)  # [batch_size * sequence_length]
                # 确保 labels 是整数类型
                labels = labels.long()
                # 计算损失
                loss = criterion(outputs, labels)
                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_duration = time.time() - epoch_start_time
            remaining_epochs = epochs - (epoch + 1)
            remaining_time = remaining_epochs * epoch_duration
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Time: {epoch_duration:.2f}s, Remaining: {remaining_time / 60:.2f}min")
        end_training_time = time.time()
        total_training_time = end_training_time - start_training_time
        print(f"Total Training Time: {total_training_time / 60:.2f}min")

    def labels_to_words(self, labels):
        '''输入标签[0,2...]，输出分割后词语位置列表[(st,ed),...]'''
        words = []
        current_start = None

        for i, label in enumerate(labels):
            if label == 0:  # B
                current_start = i
            elif label == 2:  # E
                if current_start is not None:
                    words.append((current_start, i))
                    current_start = None
            elif label == 3:  # S
                words.append((i, i))

        return words

    def evaluate_model_sentence_convert(self, vocab, label_map, test_loader):
        self.eval()
        inverse_vocab = {v: k for k, v in vocab.items()}
        inverse_label_map = {v: k for k, v in label_map.items()}

        true_words_all = []  # 所有真实词语
        pred_words_all = []  # 所有预测词语

        with torch.no_grad():   # 禁用梯度计算
            for sentences, labels in tqdm(test_loader, desc="Evaluating"):  # 显示进度条
                sentences = sentences.to(device)  # 将输入句子移动到设备
                labels = labels.to(device)  # 将标签移动到设备

                logits = self(sentences)  # (batch_size, max_len, num_tags)
                predictions = torch.argmax(logits, dim=-1)  # (batch_size, max_len)

                # 批量处理：将句子和标签转换为词语列表
                for sentence, true_labels, pred_labels in zip(sentences, labels, predictions):
                    # 将索引转换为字符和标签
                    sentence = [inverse_vocab[idx.item()] for idx in sentence if idx != 0]
                    true_labels = [inverse_label_map[idx.item()] for idx in true_labels if idx != -1]
                    pred_labels = [inverse_label_map[idx.item()] for idx in pred_labels]

                    # 转换为词语列表
                    true_words = self.labels_to_words(sentence, true_labels)
                    pred_words = self.labels_to_words(sentence, pred_labels)

                    all_predictions = []

                    for sentence, prediction in zip(sentences, predictions):
                        decoded_sentence = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in sentence if
                                            idx != 0]
                        word_positions = self._decode_word_positions(decoded_sentence, prediction, inverse_label_map)
                        all_predictions.append(word_positions)

                    true_words_all.extend(true_words)
                    pred_words_all.extend(pred_words)

                # 使用 sklearn 计算评估指标
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_words_all, pred_words_all, average='weighted', zero_division=0
            )

            # 打印评估结果
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            return all_predictions

    def evaluate_model_by_tag(self, test_loader):
        self.eval()

        true_labels_all = []  # 所有真实标签
        pred_labels_all = []  # 所有预测标签

        with torch.no_grad():  # 禁用梯度计算
            for sentences, labels in tqdm(test_loader, desc="Evaluating"):  # 显示进度条
                sentences = sentences.to(device)  # 将输入句子移动到设备
                labels = labels.to(device)  # 将标签移动到设备

                logits = self(sentences)  # (batch_size, max_len, num_tags)
                predictions = torch.argmax(logits, dim=-1)  # (batch_size, max_len)

                # 忽略填充部分 (-1)
                mask = (labels != -1)

                # 只保留有效部分
                valid_true_labels = labels[mask].cpu().numpy()
                valid_pred_labels = predictions[mask].cpu().numpy()

                true_labels_all.extend(valid_true_labels)
                pred_labels_all.extend(valid_pred_labels)

        # 使用 sklearn 计算评估指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_all, pred_labels_all, average='weighted', zero_division=0
        )

        # 打印评估结果
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def count_fit_words(self, true_labels, pred_labels):
        true_word, pred_word, correct = 0, 0, 0
        for Ttag, Ptag in zip(true_labels, pred_labels):
            if Ttag == '0' or '3':
                true_word+=1
            if Ptag == '0' or '3':
                pred_word+=1
            if (Ttag == '0' or '3') and (Ptag == '0' or '3') and (Ttag == Ptag):
                correct+=1
        return true_word, pred_word, correct

    def evaluate_model(self, test_loader, label_map):
        self.to(device)
        self.eval()

        match_count, total_true_words, total_pred_words, outputs = 0,0,0,[]

        with torch.no_grad():
            for sentences, labels in tqdm(test_loader, desc="Evaluating"):
                sentences, labels = sentences.to(device), labels.to(device)

                # 前向传播
                out=self.forward(sentences)
                pred_labels_list = torch.argmax(out, dim=-1)

                # 批量去填充
                mask = (labels != -1)
                # 提取非填充部分，保留句子分割
                valid_true = []
                valid_pred = []
                for i in range(labels.size(0)):  # 遍历每个句子
                    # 对于 labels，提取非填充部分并转换为列表
                    valid_true.append(labels[i][mask[i]].cpu().tolist())

                    # 对于 pred_labels_list，提取非填充部分并转换为列表
                    valid_pred.append(pred_labels_list[i][mask[i]].cpu().tolist())

                # 处理每个样本
                for true_label, pred_label in zip(valid_true, valid_pred):

                    true_word, pred_word, correct = self.count_fit_words(true_label, pred_label)

                    total_true_words += true_word
                    total_pred_words += pred_word
                    match_count += correct
                    outputs.append(pred_label)

        # 计算指标
        precision = match_count / total_pred_words if total_pred_words > 0 else 0
        recall = match_count / total_true_words if total_true_words > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return precision, recall, f1, outputs