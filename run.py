import time
import argparse
import pickle
from MSN import MSN
import torch
import os
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


task_dic = {
    'ubuntu':'./dataset/ubuntu_data/',
    'douban':'./dataset/DoubanConversaionCorpus/',
    'alime':'./dataset/E_commerce/'
}
data_batch_size = {
    "ubuntu": 32,
    "douban": 150,
    "alime":  200
}

## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=False,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--max_utterances",
                    default=10,
                    type=int,
                    help="The maximum number of utterances.")
parser.add_argument("--no_bert",default=False ,type=bool,help="don't use bert")
parser.add_argument("--model_load" ,default=False,type=bool,help="don't use ddddbert")
parser.add_argument("--max_words",
                    default=50,
                    type=int,
                    help="The maximum number of words for each utterance.")
parser.add_argument("--batch_size",
                    default=0,
                    type=int,
                    help="The batch size.")
parser.add_argument("--gru_hidden",
                    default=512,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--learning_rate",
                    default=0.00003,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=20,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: bert" )
parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="bert file location")
parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="bert_cache", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--do_lower_case", action='store_true',default=True,
                        help="Set this flag if you are using an uncased model.")

args = parser.parse_args()
args.batch_size = data_batch_size[args.task]
args.save_path += args.task + '.' + MSN.__name__ + ".pt"
args.score_file_path = task_dic[args.task] + args.score_file_path
#load bert



print(args)
print("Task: ", args.task)


def train_model():
    path = task_dic[args.task]
   # X_train_utterances, X_train_responses, y_train = pickle.load(file=open(path+"train.pkl", 'rb'))
    #X_dev_utterances, X_dev_responses, y_dev = pickle.load(file=open(path + "test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    #makebertinput(X_train_utterances, X_train_responses, X_dev_utterances, X_dev_responses,vocab,y_train,y_dev)


    _,_, y_train = pickle.load(file=open(path + "train.pkl", 'rb'))
#   data = pickle.load(file=open(path + "train.pkl", 'rb'))
    _, _, y_dev = pickle.load(file=open(path + "test.pkl", 'rb'))

    B_train_utterances,B_train_utterances_len, B_train_responses,B_train_responses_len = pickle.load(file=open("bert_input/train_token.pkl", 'rb'))#쪼갠다음에 합치는게 낫다 이말이야.
    B_dev_utterances,B_dev_utterances_len, B_dev_responses,B_dev_responses_len = pickle.load(file=open("bert_input/dev_token.pkl", 'rb'))



    #makebertinput(X_train_utterances, X_train_responses, X_dev_utterances, X_dev_responses,vocab,y_train)


   # B_train_utterances=B_train_utterances[:100]
    #B_train_utterances_len=B_train_utterances_len[:100]
    #B_train_responses=B_train_responses[:100]
    #B_train_responses_len=B_train_responses_len[:100]
    y_train=y_train[:100000]
    #B_dev_utterances= B_dev_utterances[:50]
    #B_dev_utterances_len=B_dev_utterances_len[:50]
    #B_dev_responses=B_dev_responses[:50]
    #B_dev_responses_len=B_dev_responses_len[:50]
    y_dev=y_dev[:50000]







    model = MSN(word_embeddings, args=args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    bert_tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)

    bert_tokenizer.add_tokens(['_eos_'])
    model.bert_model.resize_token_embeddings(len(bert_tokenizer))
    if args.model_load  is True:
        model.load_state_dict(torch.load(args.save_path))
        print("모델 로드 했다")
    model.fit(
        B_train_utterances, B_train_responses, y_train,
        B_dev_utterances, B_dev_responses, y_dev, bert_tokenizer,
        B_train_utterances_len,B_train_responses_len, B_dev_utterances_len, B_dev_responses_len
    )
'''
def makebertidx( B_train_utterances, B_train_responses,B_dev_utterances, B_dev_responses):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    bert_tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)

'''

def makebertinput(train_u,train_r,dev_u,dev_r,vocab,y_train,y_dev):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    bert_tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case,cache_dir=args.cache_dir if args.cache_dir else None)
    #tokenized_texts=tokenized_texts = [bert_tokenizer.tokenize("i am hppy")]
    #print (tokenized_texts[0])

    reverse_vocab={v:k for k, v in vocab.items()}

    train_bu=[]#총 백만.이거는 문장별 토크나이저로 나눈거임. ..
    train_word_len=[]#여기에는 문장별 토크나이저의 길이 정보가 필요함.
    for i,(context,y) in enumerate(zip(train_u,y_train)): #context len =10
        if i==100000:
            break
        context_b=[]
        context_word_len=[]
        if(i%10000==0):
            print(i)

        for utterance in context: # utterance max =50
            utterance_word_len=[]
            utterance_t=[]
            for word_idx in utterance :
                if(word_idx==0): continue
                word=reverse_vocab[word_idx]#문장 단어 수와 단어별 토큰수를 알아야해.

                token=bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word))
                utterance_word_len.append(len(token))
                utterance_t=utterance_t+token
            #print(utterance_b)
            #utterance_t+= [0 for i in range(50-len(utterance_t))]#맥스 단어가 50임 빠끄
            context_b.append(utterance_t)
            context_word_len.append(utterance_word_len)
        train_bu.append(context_b)
        train_word_len.append(context_word_len)



    train_br = []
    train_response_word_len=[]
    for i,(utterance, y)in enumerate(zip(train_r, y_train)): # utterance max =1문장
        if i==100000:
            break
        if (i % 10000 == 0):
            print(i)
        utterance_word_len = []
        utterance_t = []
        for word_idx in utterance :
            if(word_idx==0): continue
            word=reverse_vocab[word_idx]
            token=bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word))
            utterance_word_len.append(len(token))
            utterance_t=utterance_t+token
        '''
        if (len(utterance_b) == 0):#백만개에서 줄어듬......
            print("response missing!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            continue
        '''
        #utterance_t += [0 for i in range(50 - len(utterance_t))]
        train_br.append(utterance_t)
        train_response_word_len.append(utterance_word_len)

        #print(utterance_t)
    print("end")
    pickle.dump([train_bu,train_word_len,train_br,train_response_word_len],file=open("bert_input/train_token.pkl", 'wb'))
    



    dev_bu = []  # 총 백만.
    dev_word_len = []
    for i,(context,y) in enumerate(zip(dev_u,y_dev)):  # context len =10
        if i==50000:
            break
        if (i % 10000 == 0):
            print(i)
        context_b = []
        context_word_len = []
        for utterance in context:  # utterance max =50
            utterance_word_len = []
            utterance_t = []
            for word_idx in utterance:
                if (word_idx == 0): continue
                word = reverse_vocab[word_idx]
                token = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word))
                utterance_word_len.append(len(token))
                utterance_t = utterance_t + token

            # print(utterance_b)
            context_b.append(utterance_t)
            context_word_len.append(utterance_word_len)
        dev_bu.append(context_b)
        dev_word_len.append(context_word_len)


    dev_br = []
    dev_response_word_len = []
    for i,(utterance,y) in enumerate(zip(dev_r,y_dev)):  # utterance max =1문장
        if i==50000:
            break
        if (i % 10000 == 0):
            print(i)
        utterance_word_len = []
        utterance_t = []
        for word_idx in utterance:
            if (word_idx == 0): continue
            word = reverse_vocab[word_idx]
            token = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(word))
            utterance_word_len.append(len(token))
            utterance_t = utterance_t + token
        '''
        if (len(utterance_b) == 0):
            continue
        '''
        dev_response_word_len.append(utterance_word_len)
        dev_br.append(utterance_t)

    pickle.dump([dev_bu,dev_word_len, dev_br,dev_response_word_len], file=open("bert_input/dev_token.pkl", 'wb'))

    #print(train_br)


def test_model():
    path = task_dic[args.task]
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))

    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)

def test_adversarial():
    path = task_dic[args.task]
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    print("adversarial test set (k=1): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_1.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=2): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_2.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=3): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_3.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    if args.is_training:
        train_model()
      #  test_model()
    else:
        test_model()
        # test_adversarial()
    end = time.time()
    print("use time: ", (end-start)/60, " min")




