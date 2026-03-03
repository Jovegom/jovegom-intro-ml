import pandas as pd
import re
from unidecode import unidecode
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# descargar recursos si faltan (silencioso si ya están)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

URL = 'https://breathecode.herokuapp.com/asset/internal-link?id=932&path=url_spam.csv'
print('Leyendo', URL)
df = pd.read_csv(URL)

def preprocess(text):
    try:
        text = str(text).lower()
    except Exception:
        text = ''

    # extraer y conservar partes de la url si existe
    url_text = ''
    m = re.search(r"(https?://\S+|www\.\S+)", text)
    if m:
        u = m.group(0)
        u = re.sub(r"https?://", '', u)
        u = re.sub(r"^www\.", '', u)
        parts = re.split(r"[\./\?=&#_-]+", u)
        url_text = ' '.join([p for p in parts if p])

    # eliminar urls y limpiar
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [unidecode(token) for token in tokens]
    stop_words = stopwords.words('spanish')
    stop_words.append('espana')
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    result = ' '.join(tokens)
    if url_text:
        return f"{result} {url_text}".strip()
    return result

print('Preprocesando URLs...')
df['url_prepro'] = df['url'].apply(preprocess)

text = ' '.join(df['url_prepro'].dropna().astype(str).values)
if not text.strip():
    print('No hay texto en url_prepro')
else:
    print('Generando wordcloud...')
    wc = WordCloud(width=1200, height=600, background_color='white', max_words=200).generate(text)
    wc.to_file('wordcloud_url_prepro.png')
    print('Wordcloud guardado en wordcloud_url_prepro.png')
