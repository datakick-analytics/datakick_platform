import re
import nltk
import unicodedata
from bs4 import BeautifulSoup

tokenizer = nltk.tokenize.toktok.ToktokTokenizer()


# Strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    
    return stripped_text


# Remove accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    return text


# Remove special characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    
    return text


# Remove stopwords
def remove_stopwords(text):
    stopword_list = nltk.corpus.stopwords.words('english')
    
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    
    filtered_text = ' '.join(filtered_tokens)    
    
    return filtered_text


# Normalize text corpus
def normalize_corpus(corpus, 
                     html_stripping=True, accented_char_removal=True, text_lower_case=True, 
                     special_char_removal=True, stopword_removal=True):
    
    normalized_corpus = []
    
    for doc in corpus:
        
        if html_stripping:
            doc = strip_html_tags(doc)
        
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        if text_lower_case:
            doc = doc.lower()
            
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
          
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        
        if special_char_removal:
            doc = remove_special_characters(doc)  
        
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(doc)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
