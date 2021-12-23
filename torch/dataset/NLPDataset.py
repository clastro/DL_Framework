# Youtube 카테고리 분류를 위한 데이터셋 정하기
from torch.utils.data import Dataset
class YoutubeTextClassificationDataset(Dataset): #torch util의 Dataset을 Load
    def __init__(self,
               file_path = "./data/youtube_category_train.csv", #파일 경로
               num_label = 18, #라벨 개수
               device = 'gpu', #device는 GPU
               max_seq_len = 128, #512 # KoBERT max_length
               tokenizer = None #Tokenizer 종류 초기설정
               ):
        self.file_path = file_path
        self.device = device
        self.data =[]
        #self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.tokenizer = get_tokenizer()

        df = pd.read_csv(self.file_path, encoding='utf-8-sig')
        df = df.sample(frac=1,random_state=0)

        for title,topic_label in tqdm(zip(df['title'],df['topic_label'])):
            index_of_words = self.tokenizer.encode(title) #DataFrame에서 title 칼럼으로 단어 index 형성 (필요에 따라 형태소 분석)
            token_type_ids = [0] * len(index_of_words)
            attention_mask = [1] * len(index_of_words)

            # Padding Length
            padding_length = max_seq_len - len(index_of_words)
            # Zero Padding
            index_of_words += [0] * padding_length
            token_type_ids += [0] * padding_length
            attention_mask += [0] * padding_length


            # Label
            label = topic_label
            data = {
                  'input_ids': torch.tensor(index_of_words).to(self.device),
                  'token_type_ids': torch.tensor(token_type_ids).to(self.device),
                  'attention_mask': torch.tensor(attention_mask).to(self.device),
                  'labels': torch.tensor(label).to(self.device)
                 }    
            self.data.append(data)
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        item = self.data[index]
        return item
