import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader
from DialogueDataset import DialogueDataset
from Metrics import Metrics
import logging
from torch.utils.data import TensorDataset
from transformers import AdamW

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, utterlen):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.utter_len = utterlen


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.patience = 0
        self.init_clip_max_norm = 3.0# bert adam 에선 전멸임..
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        raise NotImplementedError

    def convert_examples_to_features(self,X_train_utterances, X_train_responses, tokenizer):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
       # tokens_a=tokenizer.convert_tokens_to_ids("[CLS]")

  #      for context in X_train_utterances:
 #           for utterance in context:

#        X_train_utterances,
        #label_map = {label: i for i, label in enumerate(y_train)}
        maxbertlen=256
        features = []
        for (ex_index, (utterances ,response)) in enumerate(zip(X_train_utterances,X_train_responses)):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(utterances)))

            #토큰 인덱스로 변형된거 읽어와서 CLS,SEP 끝에 붙여줌.
            tokens_a=[]
            utterlen=[]
            for utterance in utterances:
                utterlen.append(len(utterance))
                tokens_a=tokens_a+ utterance+[tokenizer.convert_tokens_to_ids("_eos_")]
             #   print(tokenizer.convert_ids_to_tokens(tokens_a))
            tokens_a = [tokenizer.cls_token_id] + tokens_a + [tokenizer.sep_token_id]
         #   if(len(response)>51): "토큰단위라 클수도 있다."
          #      print("something wrong")
            tokens_b = response+[tokenizer.sep_token_id]
            utterlen.append(len(response))
            if (len(utterlen)!=11):#문장이 아에 없을때..
                utterlen=[0]*(11-len(utterlen))+utterlen


            input_ids = tokens_a + tokens_b
            if len(input_ids)>maxbertlen:
                input_ids=[tokenizer.cls_token_id]+input_ids[-maxbertlen+1:]
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            segment_ids = [0] * (len(input_ids) - len(tokens_b))# 컨텍스트 다합친거.
            segment_ids += [1] * len(tokens_b) # #이건 리스폰스.

            if len(input_ids)>350:
                print(len(input_ids))
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = maxbertlen - len(input_ids)

            if (padding_length>0):
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                input_mask = input_mask + ([tokenizer.pad_token_id] * padding_length)
                segment_ids = segment_ids + ([tokenizer.pad_token_id] * padding_length)#패딩은 0이다.

  #          assert len(input_ids) == 256
   #         assert len(input_mask) == 256
    #        assert len(segment_ids) == 256

            #label_id=y_train[ex_index]
            #label_id = label_map[example.label]

            if ex_index < 1:
                logger.info("*** Example ***")
                logger.info("tokens_idx: %s" % " ".join(
                    [str(x) for x in input_ids]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                #logger.info("label: %d " % (utter))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              utterlen=utterlen))
        return features

    def train_step(self, i, data):
        with torch.no_grad():
            batch_ids,batch_mask,batch_seg,batch_utterlen,batch_y = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()

        logits = self.forward([batch_ids,batch_mask,batch_seg,batch_utterlen])

        loss = self.loss_func(logits, target=batch_y)
        loss.backward()
        self.optimizer.step()
        if i%10==0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),batch_y.size(0)) )  # , accuracy, corrects
        return loss


    def fit(self, X_train_utterances,  X_train_responses, y_train, ############################여기가 메인임.
                  X_dev_utterances, X_dev_responses, y_dev, tokenizer):

        if torch.cuda.is_available(): self.cuda()

        features=self.convert_examples_to_features(X_train_utterances, X_train_responses,tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_utterlen_ids = torch.tensor([f.utter_len for f in features], dtype=torch.long)#배치당 한 컨텍스트 리스폰스 세트임.f는 고로 한개의
        y_labels = torch.FloatTensor(y_train)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_utterlen_ids, y_labels)

        #dataset = DialogueDataset(X_train_utterances, X_train_responses, y_train)#아직은  인덱스인데 여기서 tensor로 바뀌는게 문제임.
        #이것도 인덱스.
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        self.loss_func = nn.BCELoss()
      #  self.optimizer =optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2_reg)

        if self.args.no_bert is True:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if 'albert_model' not in n]
                 }
            ]
            print("bert 동결 함")
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3,weight_decay=self.args.l2_reg, correct_bias=True)
        else:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.l2_reg},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            print("bert 학습중")
            self.optimizer=AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, weight_decay=self.args.l2_reg, correct_bias=True)

        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch+1, "/", self.args.epochs)
            avg_loss = 0

            self.train()
            for i, data in enumerate(dataloader):#원래 배치는 200

                loss = self.train_step(i, data)


                if i > 0 and i % 500000== 0:#200*500 십만..지금은 16
                    self.evaluate(X_dev_utterances, X_dev_responses, y_dev, tokenizer)
                    self.train()

                if epoch >= 2 and self.patience >= 1:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(self.args.save_path))
                    if self.args.no_bert is True:
                        self.adjust_learning_rate(0.6)
                    else:
                        self.adjust_learning_rate(0.95)
                    self.patience = 0

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
            cnt = len(y_train) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss/cnt))

            self.evaluate(X_dev_utterances, X_dev_responses, y_dev,tokenizer)


    def adjust_learning_rate(self, decay_rate=.8):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)


    def evaluate(self, X_dev_utterances, X_dev_responses, y_dev,tokenizer,is_test=False):
        y_pred = self.predict(X_dev_utterances, X_dev_responses,tokenizer)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, y_dev):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )

        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:",  result[3], "\t",
              "R2:",  result[4], "\t",
              "R5:",  result[5])

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + self.best_result[5]:
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:",  self.best_result[3], "\t",
                  "R2:",  self.best_result[4], "\t",
                  "R5:",  self.best_result[5])
            self.patience = 0
            self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1


    def predict(self, X_dev_utterances, X_dev_responses,tokenizer):
        self.eval()
        y_pred = []
        features = self.convert_examples_to_features(X_dev_utterances, X_dev_responses, tokenizer)

  #      for f in features:
   #         print(f.input_ids)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_utterlen_ids = torch.tensor([f.utter_len for f in features],dtype=torch.long)  # 배치당 한 컨텍스트 리스폰스 세트임.f는 고로 한개의
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_utterlen_ids)#여기도 만개로 수정했음.
        #dataset = DialogueDataset(X_dev_utterances, X_dev_responses)
        dataloader = DataLoader(dataset, batch_size=128)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_ids, batch_mask, batch_seg, batch_utterlen= (item.cuda() for item in data)
            with torch.no_grad():
                logits = self.forward([batch_ids, batch_mask, batch_seg, batch_utterlen])
            if i % 10==0:
                print('Batch[{}] batch_size:{}'.format(i, batch_ids.size(0)))  # , accuracy, corrects
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred


    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()

