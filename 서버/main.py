# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 패키지 가져오기
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# 파이썬 모델 실행 관련 패키지
import re
import json
import numpy as np


# LSTM 모델 관련 패키지
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from kiwipiepy import Kiwi


# DB 관련 패키지
import sqlalchemy
from sqlalchemy import create_engine


# 분류 관련 패키지
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


# 파이썬 서버 실행 관련 패키지
from fastapi import FastAPI
from pydantic import BaseModel





# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 클래스 선언
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# API로 전송받을 메시지 데이터 형태
class ChatMsgData(BaseModel):
    ChannelID: str
    SessionID: str
    OrderNum: int
    User_ID: str
    Disp_Name: str
    ChatTime: str
    ChatMsg: str


# 데이터 셋에 대한 클래스
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))  


# BERT 분류기
class BERTClassifier(nn.Module): ## 클래스를 상속
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=4,   ##클래스 수 조정##
                 dr_rate=0.5,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)




# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# DB 연결하기
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# DB 로그인 관련 상수
Guser = "root"
Gpassword = "1234"
Ghost = "localhost"
Gport = "3306"
informs = {"user": Guser, "password":Gpassword, "host":Ghost,"port":Gport}


# 스트리머 이름을 가진 SCHEMA가 존재하는지 확인하는 함수
def CheckSchemaExist(StreamerID):
    # 존재하지 않을 경우 DB에 해당 SCHEMA와 ViewerInfo TABLE을 생성합니다.
    # DB를 RETURN 합니다.
    # print("[Python]: CheckSchema Exist 시작")

    db_url = f'mysql://{informs["user"]}:{informs["password"]}@{informs["host"]}:{informs["port"]}'
    engine = create_engine(url=db_url)
    # print("[Python]: Port에 연결되었습니다.")

    engine.connect().execute(sqlalchemy.text(f'CREATE DATABASE IF NOT EXISTS {StreamerID}'))
    db_url = f'mysql://{informs["user"]}:{informs["password"]}@{informs["host"]}:{informs["port"]}/{StreamerID}'
    engine = create_engine(url=db_url)
    # print("[Python]: Streamer ID로 한 DB 주소를 설정")

    engine.connect().execute(sqlalchemy.text('''CREATE TABLE IF NOT EXISTS ViewerInfo (
                                                User_ID VARCHAR(60) PRIMARY KEY,
                                                TotalScore FLOAT DEFAULT 0.0
                                                );
                                                '''))
    engine.connect().commit()
    return engine.connect()


# 시청자의 누적 총점수를 가져오는 함수입니다.
def GetViewerTotalScore(UserID, engine):
    # UserID 를 가진 row 가 있는지 먼저 확인을 하고
    # 있으면 가져오고 없으면 만들어서 가져옵니다.
    engine.execute(sqlalchemy.text(f'''
                                    INSERT IGNORE INTO ViewerInfo (User_ID, TotalScore)
                                    VALUES ("{UserID}", 0.0);
    '''))
    engine.commit()

    Result = engine.execute(sqlalchemy.text(f'''SELECT TotalScore
                                                FROM ViewerInfo
                                                WHERE User_ID = "{UserID}";
    ''')).fetchall()[0][0]
    return Result


# DB에 Session Chat Log Table이 없을 시 Table을 생성합니다.
def CheckTableExist(TableName, DBEngine):
    # print("[Python]: Table이 존재하는지 확인합니다.")
    DBEngine.execute(sqlalchemy.text(f'''CREATE TABLE IF NOT EXISTS {TableName} (
                                        User_ID VARCHAR(60),
                                        Disp_Name VARCHAR(100),
                                        ChatTime VARCHAR(60) PRIMARY KEY,
                                        Score FLOAT,
                                        ChatMsg VARCHAR(200)
                                        );
                                        '''))
    DBEngine.commit()
    # print("[Python]: Table 존재 확인.")
    DBEngine.execute(sqlalchemy.text(f'''CREATE OR REPLACE view {TableName}_view
                                        AS
                                        SELECT User_ID, Sum(Score) AS SessionScore
                                        FROM {TableName}
                                        GROUP BY User_ID;
                                        '''))
    DBEngine.commit()
    # print("[Python]: View 존재 확인")


# ViewerInfoTable에 시청자의 갱신된 정보를 넣습니다.
def UpdateViewerInfo(UserID, Score, DBEngine):
    DBEngine.execute(sqlalchemy.text(f'''INSERT INTO ViewerInfo (User_ID, TotalScore)
                                        VALUES ("{UserID}",{Score})
                                        ON DUPLICATE KEY UPDATE
                                            User_ID = "{UserID}",
                                            TotalScore = {Score};
    '''))
    DBEngine.commit()

# SessionID TABLE (메시지 로그)에 메시지 로그 정보를 추가합니다.
def AddMsgLog(SessionID, UserID, Disp_Name, ChatTime, Score, ChatMsg, DBEngine):
    DBEngine.execute(sqlalchemy.text(f'''INSERT IGNORE INTO {SessionID}
                                            (User_ID, Disp_Name, ChatTime, Score, ChatMsg)
                                        VALUES
                                            ("{UserID}", "{Disp_Name}", "{ChatTime}", {Score}, "{ChatMsg}");
    '''))
    DBEngine.commit()

# View에서 누적 점수를 가져옵니다.
def GetViewerSessionScore(UserID, SessionName, DBEngine):
    Result = DBEngine.execute(sqlalchemy.text(f'''SELECT SessionScore
                                        FROM {SessionName}_view
                                        WHERE User_ID = "{UserID}";
    ''')).fetchall()[0][0]

    return Result





# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 점수 예측 모델
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
LSTM_Model = load_model('./models/LSTM.hdf5')
LSTM_preprocess_configuration = json.load(open('./models/data_configs.json','r'))
LSTM_word_dic = LSTM_preprocess_configuration['vocab']
LSTM_MAX_SEQ_LEN = LSTM_preprocess_configuration['MAX_SEQ_LEN']

LSTM_tokenizer = Tokenizer()
LSTM_tokenizer.word_index = LSTM_word_dic

kiwi = Kiwi(model_type='sbg', typos=None)

kiwi.load_user_dictionary('./user_dic.txt')
kiwi.prepare()

def extract(text):
    result = kiwi.analyze(text)
    result = [(token, pos) for token, pos, _, _ in result[0][0]]
    
    return result

def preprocessing(review):
    # 한글, 영문자, 숫자 등의 문자만 추출합니다.
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ0-9a-zA-Z.\\s]", "", review)    
    
    # kiwi 형태소 분석을 합니다.  
    T = list(extract(review_text))
            
    # 유의미한 형태소만을 추출합니다. (조사 제외)
    PosList = []
    for tuples in T:
        if tuples[1][0] != 'J' and tuples[1] != 'EF':          
            PosList.append(tuples[0])
            
    # 분석에 필요한 형태소의 모임만을 return 합니다.
    return PosList


# 혐오사전 로드
HyumODict = json.load(open('./models/aversion_dic.json','r'))

def HyumOCheck(Inputs):
    DictScore = 0
    for item in Inputs:
        if item in HyumODict:
            DictScore += HyumODict[item]
    
    return DictScore

def LSTM_Predict(Inputs):
    sentence = preprocessing(Inputs)
    DictScore = HyumOCheck(sentence)
    Vec_S = LSTM_tokenizer.texts_to_sequences([sentence])
    Pad_S = pad_sequences(Vec_S, maxlen = LSTM_MAX_SEQ_LEN)
    Result = LSTM_Model.predict(Pad_S)[0][0]

    # [[결과값]]
    return Result, DictScore






# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 분류 예측 모델
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# Pytorch에서 사용할 프로세서
device = torch.device("cpu")
bertmodel, vocab = get_pytorch_kobert_model()


# BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load('./models/model_state_dict.pt',map_location=torch.device('cpu')))
classLabel = [0,1.0,1.1,1.2]


# 파라미터
max_len = 64
batch_size = 64


# 토크나이저 가져오기
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


# 예측 함수
def predict(sentence):
    data = [sentence, '0']
    dataset_another = [data]
    logits = 0
    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits)

    return classLabel[logits]





# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 서버
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

app = FastAPI()

@app.get('/')
async def root():
    return {"message": "Hello, this is Python Server"}



@app.post("/analyze/")
async def create_item(item: ChatMsgData):
    # print("[Python]: 메시지를 받았습니다.")
    # 넘겨받은 데이터를 딕셔너리화 한다.
    Msg_dict = item.dict()
    # print("[Python]: 딕셔너리화 했습니다.")

    # 채팅 메시지는 Msg_Sentence 변수에 저장해놓는다.
    Msg_Sentence = Msg_dict["ChatMsg"]
    # print("[Python]: Chat Msg를 변수에 저장했습니다.")

    # DBEngine (스키마가 있는지 확인해 있으면 가져오고 없으면 생성한다.)
    DBEngine = CheckSchemaExist(Msg_dict["ChannelID"])
    # print("[Python]: 스키마를 가져왔습니다.")

    # 시청자의 누적 총점수를 가져온다.
    ViewerTotalScore = GetViewerTotalScore(Msg_dict["User_ID"], DBEngine)
    # print("[Python]: 누적 총점수를 가져왔습니다.")

    # KoBERT로 예측한 심각도 배율
    PredictedMult = float(predict(Msg_Sentence))
    # print(f"[Python]: 분류 배율이 나왔습니다. {PredictedMult}")
    # print(f'[Python]: PredictedMult = {PredictedMult}, Type = {type(PredictedMult)}')

    # 일차적 분류 배율이 0이 아닐 때 (비악플이 아닐 때) 추가 조치를 들어간다.
    if PredictedMult > 0:

        # LSTM으로 점수를 계산하고 혐오단어사전을 검색한다.
        JumSoo, DictScore = LSTM_Predict(Msg_Sentence)
        # print(f"[Python]: LSTM 점수가 나왔습니다. {JumSoo}")
        # print(f"[Python]: 혐오사전 점수가 나왔습니다. {DictScore}")
        # print(f'[Python]: JumSoo = {JumSoo}, Type = {type(JumSoo)}')

        # 만일 LSTM으로 예측한 점수가 0점보다 작다면 0점으로 계산한다.
        if JumSoo < 0:
            JumSoo = 0

        # LSTM 점수 x KoBERT 심각도 분류(배율)
        SentenceScore = float(JumSoo*PredictedMult)
        # print(f'[Python]: InitValue = {InitValue}, Type = {type(InitValue)}')

        # 혐오단어사전 점수를 더한다.
        SentenceScore += DictScore

    # 일차적 분류 배율이 0이라면 (비악플이라면) 점수는 -5점으로 고정된다.
    else:
        # 선플일 경우 음수를 기본값으로 한다.
        # 시청자의 세션 누적 점수를 가져온다.
        ViewerSessionScore = GetViewerSessionScore(Msg_dict["User_ID"],Msg_dict["SessionID"], DBEngine)
        # print("[Python]: 세션 누적 점수 가져왔습니다.")
        # 50점 이상의 고 위험군의 경우 선플을 쓸 때마다 1점씩 감점한다.
        if ViewerSessionScore >= 50:
            SentenceScore = -1.0
        # 20점 이상의 저 위험군의 경우 선플을 쓸 때마다 3점씩 감점한다.
        elif ViewerSessionScore >= 20:
            SentenceScore = -3.0
        # -40점 (선플 스택 10개)일 경우 더이상 점수가 아래로 내려가지 않습니다.
        elif ViewerSessionScore <= 40:
            SentenceScore = 0.0
        # 기타 경우 오진의 가능성이 있어 4점씩 감점한다.
        else:
            SentenceScore = -4.0
    # print(f"[Python]: 문장 점수는 {SentenceScore}입니다.")

    # 누적 총점수에 더한다.
    ViewerTotalScore += SentenceScore

    # 누적 총점수를 UPDATE 한다.
    UpdateViewerInfo(Msg_dict["User_ID"], ViewerTotalScore, DBEngine)
    # print("[Python]: 누적 총점수 UPDATE 완료")

    # 채팅 메시지 로그 Table이 존재하는지 확인한다.
    CheckTableExist(Msg_dict["SessionID"], DBEngine)
    # print("[Python]: 채팅 메시지 로그 Table 존재 확인")

    # 채팅 메시지 로그 Table을 UPDATE 한다.
    AddMsgLog(Msg_dict["SessionID"],
    Msg_dict["User_ID"],
    Msg_dict["Disp_Name"],
    Msg_dict["ChatTime"],
    SentenceScore,
    Msg_dict["ChatMsg"],
    DBEngine)
    # print("[Python]: 채팅 메시지 로그 Table Update")

    # 업데이트된 시청자의 세션 누적 점수를 가져온다.
    ViewerSessionScore = GetViewerSessionScore(Msg_dict["User_ID"],Msg_dict["SessionID"], DBEngine)
    # print("[Python]: 세션 누적 점수 가져왔습니다.")

    # 세션 누적 점수에 따라 처리결과를 달리 한다.
    if ViewerSessionScore > 125 and Msg_dict["User_ID"] != "UCy2xB_uwKU_l7o5AqwPtAHA":
        BunRyu = "BANN"
    elif ViewerSessionScore > 50 and Msg_dict["User_ID"] != "UCy2xB_uwKU_l7o5AqwPtAHA":
        BunRyu = "WARN"
    elif ViewerSessionScore > 20 and Msg_dict["User_ID"] != "UCy2xB_uwKU_l7o5AqwPtAHA":
        BunRyu = "ADVS"
    else:
        BunRyu = "NADA"

    # 선플일 경우 조치하지 않는다.
    if SentenceScore <= 0:
        BunRyu = "NADA"
    
    DBEngine.close()

    # Msg_dict에 추가정보를 더한다.
    Msg_dict.update({"Label":BunRyu,
                    "Score":SentenceScore,
                    "ViewerScore":ViewerSessionScore,
                    "TotalScore":ViewerTotalScore})

    print(f"[Python]: 문장 분석 완료\n{Msg_dict['Disp_Name']}:'{Msg_Sentence}'\n점수:{SentenceScore}\n처리결과:{BunRyu}")
    # 보낸다.
    return Msg_dict