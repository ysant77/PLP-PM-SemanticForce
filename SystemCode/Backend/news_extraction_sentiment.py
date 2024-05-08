#!pip install GoogleNews
# !pip install newspaper3k
# !pip install vaderSentiment
# import GoogleNews
# Third-party Libraries
from GoogleNews import GoogleNews
import pandas as pd
import newspaper # library to extract text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Native Libraries
from datetime import datetime
import requests
import copy


def extract_article_text(url):
    try:
        article = newspaper.Article(url=url)
        article.download()
        article.parse()
        return article.text
    except:
        return 'URL cannot be opened or read'


def get_sentiment_label(sentiment_scores):
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.52:
        return 'Positive'
    elif compound_score <= -0.48:
        return 'Negative'
    else:
        return 'Neutral'

def add_sentiment_analysis(df, text_column='article_text', sentiment_column='sentiment'):

    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = df[text_column].apply(analyzer.polarity_scores)  # Apply analyzer directly
    df[sentiment_column] = sentiment_scores
    df['sentiment_label'] = df[sentiment_column].apply(get_sentiment_label)  # Assign labels

    return df



#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




def get_vector(sentence, model, tokenizer):
    if not sentence:
        sentences=['']
    else:
        sentences=[sentence]
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings[0]


def get_similarity(df):
    # calculate the max similarity with earlier news
    length=len(df)
    unique=[]

    df_date = df.date.apply(lambda x: datetime.strptime(x, '%b %d'))
    news2datetime=dict(zip(df.index, df_date))
    if length<=1:
        return
    else:
        for i in range(1,length):
            max_sim=0
            for j in range (0,i):
                if abs(news2datetime[i]-news2datetime[j]).days<14:
                    curr_sim=cosine_similarity(df.loc[i]['vector'].reshape(1,-1),df.loc[j]['vector'].reshape(1,-1))[0][0]
                    max_sim=max(max_sim,curr_sim)
                    # below code for manual check
#                     if curr_sim>0.9:
#                         print(i, j)
#                         print(df.loc[i]['article_text'])
#                         print('====')
#                         print(df.loc[j]['article_text'])
            if max_sim<0.9:

                unique.append(i)
    df2=df.loc[unique]
    df2.reset_index(inplace=True)
    return df2


def news_extract(output_path, company_name, start_date, end_date):
    # Generation of DataFrame
    gn = GoogleNews(lang='en', region='US', encode='utf-8')

    # input company name & duration here

    # company_name = 'Tesla Inc'
    # start_date='01/07/2024' # MM/DD/YYYY
    # end_date='01/08/2024'
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%m/%d/%Y")

    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%m/%d/%Y")

    start_date_str = datetime.strptime(start_date, '%m/%d/%Y').strftime('%d_%b_%y')
    end_date_str = datetime.strptime(end_date, '%m/%d/%Y').strftime('%d_%b_%y')

    output_file_name=f"{output_path}/{company_name}_{start_date_str}_{end_date_str}.csv"

    datetime_list=pd.date_range(start_date,end_date,freq='d').to_list()
    date_list=[d.strftime('%m/%d/%Y') for d in datetime_list]
    n_date=len(date_list)

    fail_count=0

    for i in range(n_date-1):

        gn.set_time_range(start_date,end_date)
        gn.get_news(company_name)
        # Limit the number of retrieved news articles to 50
        results = gn.results(sort=True)[:50]

    # Add "http://" to each link
    df=pd.DataFrame(results)
    df['link'] = 'http://' + df['link']

    # Add a column for extracted text
    df['article_text'] = df['link'].apply(extract_article_text)

    # # Add a column to apply the NER function (optional)
    # df['extracted organizations'] = df['article_text'].apply(lambda text: get_org(text))

    # Add sentiment analysis with sentiment labels
    df = add_sentiment_analysis(df.copy())



    # Load model from HuggingFace Hub
    tokenizer_mpnet = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model_mpnet = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


    df.dropna(subset='article_text',inplace=True)
    df = df[df['article_text'] != 'URL cannot be opened or read']
    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(inplace=True)
    df['vector']=df['article_text'].apply(lambda x: get_vector(x, model_mpnet, tokenizer_mpnet))
    # df.to_csv('test_sentiment.csv', index=False)
    # deduplicate
    df2=get_similarity(copy.deepcopy(df))

    df2.to_csv(output_file_name, index=False)

    return df2, output_file_name