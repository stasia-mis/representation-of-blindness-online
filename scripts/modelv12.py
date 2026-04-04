"""
ВНИМАНИЕ НАСТРОЙКИ В КОДЕ ПОДХОДЯТ ТОЛЬКО ДЛЯ RAM 32
Для работы с большими данными:
1. Sampling strategy для быстрой разработки
2. MiniBatch алгоритмы для обучения
3. Reduced feature space
4. Incremental processing
5. Memory-efficient sparse matrices
6. Parallel processing
"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
import pymorphy2
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score
from scipy.stats import mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()
tqdm.pandas()

print("="*60)
print("АНАЛИЗ ДЛЯ БОЛЬШИХ ДАННЫХ")
print("="*60)

# ============================================================================
# КОНФИГУРАЦИЯ ДЛЯ БОЛЬШИХ ДАННЫХ
# ============================================================================

class BigDataConfig:
    """Конфигурация для работы с большими корпусами"""
    
    # SAMPLING STRATEGY
    USE_SAMPLING = False   # Использовать ли выборку для разработки
    SAMPLE_SIZE_NKO = 200000  # Размер выборки для НКО
    SAMPLE_SIZE_BLOGS = 200000  # Размер выборки для блогов
    STRATIFY_BY = 'victim_label'  # Стратификация при sampling
    
    # FEATURE ENGINEERING
    MAX_FEATURES_TFIDF = 50000  # Уменьшить для памяти например 2000
    NGRAM_RANGE = (1, 3)  # Только триграммы, можно для оптимизации поставить униграммы (1,1)
    MIN_DF = 5  # Минимальная частота документа (фильтр редких слов)
    MAX_DF = 0.7  # Максимальная доля документов (фильтр частых слов)
    
    # MODEL CONFIGURATION
    USE_SIMPLE_ENSEMBLE = True  # Упрощенный ансамбль вместо stacking
    N_ESTIMATORS = 300  # Можно поставить 100-200
    USE_MINIBATCH = True  # Использовать MiniBatch алгоритмы
    
    # LDA CONFIGURATION
    LDA_SAMPLE_SIZE = 91358  # полный корпус слишком медленный, для оптимизации лучше уменьшить 
    LDA_NUM_TOPICS = 10  # Фиксированное число тем
    LDA_PASSES = 10  # Для оптимизации лучше поставить 5
    
    # PROCESSING
    CHUNK_SIZE = 10000  # Размер чанка для инкрементальной обработки
    N_JOBS = -1  # Использовать все ядра CPU
    
    # OUTPUT
    SAVE_PREDICTIONS = False  # Не сохранять все предсказания (экономия места)
    SAVE_SAMPLE_PREDICTIONS = 2000  # Сохранить только выборку

config = BigDataConfig()

# ============================================================================
# ТЕКСТОВАЯ ПРЕДОБРАБОТКА (оптимизированная)
# ============================================================================

class FastTextPreprocessor:
    """Быстрая предобработка для больших объемов"""
    
    def __init__(self, morph_analyzer, stop_words, use_lemmatization=True):
        self.morph = morph_analyzer
        self.stop_words = stop_words
        self.use_lemmatization = use_lemmatization
    
    def clean_text(self, text):
        """Базовая очистка"""
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^а-яё\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def preprocess_to_string(self, text):
        """Полный пайплайн с опциональной лемматизацией"""
        cleaned = self.clean_text(text)
        
        if not self.use_lemmatization:
            # Быстрый режим: только очистка и удаление стоп-слов
            tokens = [w for w in cleaned.split() if w not in self.stop_words and len(w) > 2]
            return " ".join(tokens)
        
        # Полная лемматизация (медленнее)
        tokens = cleaned.split()
        lemmas = []
        for word in tokens:
            if word not in self.stop_words and len(word) > 2:
                parsed = self.morph.parse(word)[0]
                lemmas.append(parsed.normal_form)
        return " ".join(lemmas)
    
    def preprocess_batch(self, texts, show_progress=True):
        """Пакетная обработка с прогрессом"""
        if show_progress:
            return [self.preprocess_to_string(t) for t in tqdm(texts, desc="Preprocessing")]
        return [self.preprocess_to_string(t) for t in texts]

# Создать препроцессор
# Для большого количества строк: отключаем лемматизацию для скорости (или делаем batch processing)
preprocessor = FastTextPreprocessor(morph, stop_words, use_lemmatization=True)
print(f"Preprocessor: lemmatization={'ON' if preprocessor.use_lemmatization else 'OFF (fast mode)'}")

# ============================================================================
# УЛУЧШЕННЫЙ WEAK SUPERVISION (упрощенный для скорости)
# ============================================================================

class FastWeakLabeler:
    """Упрощенная версия weak labeling для больших данных"""
    
    def __init__(self):
        self.marker_dicts = {
            'victim': ['страдать', 'нуждаться', 'инвалид', 'поддержка', 
                      'благотворительность', 'трудность', 'беспомощный', 'зависимый', 'ограничен', 'недостаток',
                      'пожертвование', 'нужда', 'тяжелое положение'],
            'supercrip': ['вопреки', 'несмотря', 'преодолел', 'преодолела', 'преодолели',
                'смог', 'смогла', 'смогли', 'добился', 'добилась', 'добились',
                'достиг', 'достигла', 'достигли', 'победил', 'победила',
                'сила духа', 'пример мужества', 'герой', 'подвиг',
                'невозможное возможно', 'доказал', 'доказала',
                'вдохновляет', 'восхищает', 'удивительная история',
                'не приговор', 'несмотря ни на что', 'вопреки всему',
                'слепой но', 'незрячий но', 'инвалид но'
            ],
            'agency': ['работать', 'учиться', 'делать', 'создавать', 'самостоятельный',
                      'независимый', 'активный', 'решать', 'выбирать', 'планировать', 'развивать',
                      'организовывать', 'управлять', 'вести', 'заниматься', 'достигать', 'стремиться', 'реализовывать',  'проект', 'бизнес', 'карьера', 'цель', 'инициатива' ],
            'first_person': ['я', 'мой', 'мне', 'мы', 'наш', 'нам']
        }
    
    def score_category(self, text, category):
        """Быстрый подсчет без морфологии (для скорости)"""
        if not isinstance(text, str):
            return 0.0
        
        text_lower = text.lower()
        markers = self.marker_dicts.get(category, [])
        
        score = sum(1.0 for marker in markers if marker in text_lower)
        return score
    
    def create_labels(self, df, threshold=1.0):
        """Создать метки"""
        for cat in ['victim', 'supercrip', 'agency', 'first_person']:
            df[f'{cat}_score'] = df['text'].apply(lambda x: self.score_category(x, cat))
            df[f'{cat}_label'] = (df[f'{cat}_score'] >= threshold).astype(int)
        return df

labeler = FastWeakLabeler()

# ============================================================================
# ЗАГРУЗКА И SAMPLING
# ============================================================================

def load_with_sampling(nko_path, blogs_path, config):
    """
    Загрузка данных с опциональным sampling
    """
    print("\n" + "="*60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*60)
    
    # Загрузка
    print("Loading files...")
    nko = pd.read_csv(nko_path, engine="python", sep=",", quotechar='"', on_bad_lines="skip")
    blogs = pd.read_csv(blogs_path, engine="python", sep=",", quotechar='"', on_bad_lines="skip")
    
    # Стандартизация
    for df in [nko, blogs]:
        df.columns = [col.strip().lower() for col in df.columns]
        if 'текст' in df.columns:
            df.rename(columns={'текст': 'text'}, inplace=True)
    
    print(f"Full dataset: {len(nko)} NKO posts, {len(blogs)} blog posts")
    print(f"TOTAL: {len(nko) + len(blogs)} posts")
    
    # Добавить source
    nko['source'] = 'nko'
    blogs['source'] = 'blogs'
    
    # SAMPLING для разработки
    if config.USE_SAMPLING:
        print(f"\n⚠️  SAMPLING MODE: Using {config.SAMPLE_SIZE_NKO} NKO + {config.SAMPLE_SIZE_BLOGS} blogs")
        print("Set config.USE_SAMPLING = False to use full dataset")
        
        # Создать временные метки для стратификации
        print("Creating temporary labels for stratified sampling...")
        nko_temp = labeler.create_labels(nko.copy(), threshold=1.0)
        blogs_temp = labeler.create_labels(blogs.copy(), threshold=1.0)
        
        # Stratified sampling
        if config.SAMPLE_SIZE_NKO < len(nko):
            nko_sample = nko_temp.sample(
                n=min(config.SAMPLE_SIZE_NKO, len(nko)),
                random_state=42
            )
        else:
            nko_sample = nko_temp
        
        if config.SAMPLE_SIZE_BLOGS < len(blogs):
            blogs_sample = blogs_temp.sample(
                n=min(config.SAMPLE_SIZE_BLOGS, len(blogs)),
                random_state=42
            )
        else:
            blogs_sample = blogs_temp
        
        nko = nko_sample
        blogs = blogs_sample
        
        print(f"Sampled: {len(nko)} NKO posts, {len(blogs)} blog posts")
    
    return nko, blogs

# ============================================================================
# БЫСТРАЯ ПРЕДОБРАБОТКА
# ============================================================================

def fast_preprocess(nko, blogs, preprocessor):
    """Быстрая предобработка с прогрессом"""
    print("\n" + "="*60)
    print("ПРЕДОБРАБОТКА ТЕКСТА")
    print("="*60)
    
    # Предобработка
    print("\nProcessing NKO texts...")
    nko['text_clean'] = preprocessor.preprocess_batch(nko['text'].values)
    
    print("Processing blog texts...")
    blogs['text_clean'] = preprocessor.preprocess_batch(blogs['text'].values)
    
    # Weak labels
    print("\nCreating weak labels...")
    nko = labeler.create_labels(nko, threshold=1.0)
    blogs = labeler.create_labels(blogs, threshold=1.0)
    
    return nko, blogs

# ============================================================================
# УПРОЩЕННЫЙ ENSEMBLE ДЛЯ БОЛЬШИХ ДАННЫХ
# ============================================================================

class SimplifiedEnsemble:
    """
    Упрощенный ансамбль для больших данных:
    - 2 модели 
    - MiniBatch SGD для линейной модели
    - Simplified RF
    """
    
    def __init__(self, config):
        self.config = config
        self.vectorizer = None
        self.model = None
    
    def create_ensemble(self):
        """Создать упрощенный ансамбль"""
        if self.config.USE_MINIBATCH:
            # SGDClassifier - быстрый для больших данных
            sgd = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                max_iter=1000,
                random_state=42,
                n_jobs=self.config.N_JOBS,
                class_weight='balanced'
            )
        else:
            sgd = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=self.config.N_JOBS,
                class_weight='balanced'
            )
        
        # Упрощенный Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=10,  # Ограничена глубина
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=self.config.N_JOBS
        )
        
        # Voting вместо stacking (проще и быстрее)
        from sklearn.ensemble import VotingClassifier
        voting = VotingClassifier(
            estimators=[('sgd', sgd), ('rf', rf)],
            voting='soft',
            n_jobs=self.config.N_JOBS
        )
        
        # Multi-label wrapper
        multi_label = MultiOutputClassifier(voting, n_jobs=self.config.N_JOBS)
        
        return multi_label
    
    def fit(self, df, target_cols):
        """Обучить модель"""
        print("\nPreparing features...")
        
        # TF-IDF с ограниченными параметрами
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.MAX_FEATURES_TFIDF,
            ngram_range=self.config.NGRAM_RANGE,
            min_df=self.config.MIN_DF,
            max_df=self.config.MAX_DF
        )
        
        X = self.vectorizer.fit_transform(df['text_clean'])
        y = df[target_cols].values
        
        print(f"Feature matrix: {X.shape}")
        print(f"Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
        print(f"Memory usage: ~{X.data.nbytes / 1024 / 1024:.1f} MB")
        
        print("\nTraining ensemble...")
        self.model = self.create_ensemble()
        self.model.fit(X, y)
        
        return self
    
    def predict(self, df):
        """Предсказать"""
        X = self.vectorizer.transform(df['text_clean'])
        return self.model.predict(X)

# ============================================================================
# СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ 
# ============================================================================

def fast_statistical_comparison(nko, blogs):
    """Быстрое статистическое сравнение"""
    print("\n" + "="*60)
    print("СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ")
    print("="*60)
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    results = []
    
    for cat in categories:
        nko_scores = nko[f'{cat}_score'].values
        blog_scores = blogs[f'{cat}_score'].values
        
        # Mann-Whitney U test
        statistic, p_value = mannwhitneyu(nko_scores, blog_scores, alternative='two-sided')
        
        nko_mean = nko_scores.mean()
        blog_mean = blog_scores.mean()
        
        results.append({
            'category': cat,
            'nko_mean': nko_mean,
            'blogs_mean': blog_mean,
            'diff': blog_mean - nko_mean,
            'p_value': p_value,
            'significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Chi-square
    print("\n" + "="*60)
    print("CHI-SQUARE TESTS")
    print("="*60)
    
    combined = pd.concat([nko, blogs], ignore_index=True)
    
    for cat in categories:
        contingency = pd.crosstab(combined['source'], combined[f'{cat}_label'])
        
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            print(f"\n{cat}: Insufficient data")
            continue
        
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        print(f"\n{cat}: χ² = {chi2:.4f}, p = {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
    
    return df_results

# ============================================================================
# SIMPLIFIED LDA 
# ============================================================================

def fast_lda(df, sample_size, name):
    """Быстрый LDA на выборке"""
    print(f"\nLDA for {name} (sample: {sample_size})...")
    
    # Взять выборку если данных много
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # Токенизация
    texts = [text.split() for text in df_sample['text_clean'].values]
    
    # LDA
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    lda_model = models.LdaModel(
        corpus,
        num_topics=config.LDA_NUM_TOPICS,
        id2word=dictionary,
        passes=config.LDA_PASSES,
        random_state=42
    )
    
    print(f"Topics for {name}:")
    for idx, topic in lda_model.print_topics(num_words=8):
        print(f"  Topic {idx}: {topic}")

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главный пайплайн для больших корпусов"""
    import os
    os.makedirs('output', exist_ok=True)
    
    print("\n📊 КОНФИГУРАЦИЯ:")
    print(f"  Sampling: {config.USE_SAMPLING}")
    if config.USE_SAMPLING:
        print(f"  Sample size: {config.SAMPLE_SIZE_NKO} NKO + {config.SAMPLE_SIZE_BLOGS} blogs")
    print(f"  TF-IDF max features: {config.MAX_FEATURES_TFIDF}")
    print(f"  Lemmatization: {preprocessor.use_lemmatization}")
    print(f"  MiniBatch mode: {config.USE_MINIBATCH}")
    
    # 1. Загрузка
    print("\n[1/6] Loading data...")
    nko, blogs = load_with_sampling("nko.csv", "blogs.csv", config)
    
    # 2. Предобработка
    print("\n[2/6] Preprocessing...")
    nko, blogs = fast_preprocess(nko, blogs, preprocessor)
    
    # 3. Статистика
    print("\n[3/6] Statistical comparison...")
    stat_results = fast_statistical_comparison(nko, blogs)
    
    # 4. Обучение
    print("\n[4/6] Training model...")
    combined = pd.concat([nko, blogs], ignore_index=True)
    target_cols = ['victim_label', 'supercrip_label', 'agency_label', 'first_person_label']
    
    print(f"\nLabel distribution:")
    for col in target_cols:
        pct = (combined[col].sum() / len(combined)) * 100
        print(f"  {col}: {combined[col].sum()} ({pct:.1f}%)")
    
    # Train/test split
    train_df, test_df = train_test_split(combined, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    
    ensemble = SimplifiedEnsemble(config)
    ensemble.fit(train_df, target_cols)
    
    # 5. Evaluation
    print("\n[5/6] Evaluation...")
    y_test = test_df[target_cols].values
    y_pred = ensemble.predict(test_df)
    
    print("\nTest Metrics:")
    print(f"  Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
    print(f"  Micro F1: {f1_score(y_test, y_pred, average='micro'):.4f}")
    print(f"  Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    # 6. LDA
    print("\n[6/6] Running LDA...")
    fast_lda(nko, config.LDA_SAMPLE_SIZE, "NKO")
    fast_lda(blogs, config.LDA_SAMPLE_SIZE, "BLOGS")
    
    # Сохранение
    print("\nSaving results...")
    stat_results.to_csv('output/statistical_comparison.csv', index=False)
    
    # Сохранить только выборку предсказаний (экономия места)
    if config.SAVE_SAMPLE_PREDICTIONS > 0:
        sample_nko = nko.head(config.SAVE_SAMPLE_PREDICTIONS)
        sample_blogs = blogs.head(config.SAVE_SAMPLE_PREDICTIONS)
        
        nko_preds = ensemble.predict(sample_nko)
        blogs_preds = ensemble.predict(sample_blogs)
        
        for i, col in enumerate(target_cols):
            sample_nko[f'{col}_pred'] = nko_preds[:, i]
            sample_blogs[f'{col}_pred'] = blogs_preds[:, i]
        
        sample_nko.to_csv('output/nko_sample_predictions.csv', index=False)
        sample_blogs.to_csv('output/blogs_sample_predictions.csv', index=False)
        print(f"Saved sample predictions ({config.SAVE_SAMPLE_PREDICTIONS} posts each)")
    
    print("\n" + "="*60)
    print("✅ WW АНАЛИЗ ЗАВЕРШЕН")
    print("="*60)
    
    if config.USE_SAMPLING:
        print("\n⚠️  ВАЖНО: Использовалась ВЫБОРКА данных!")
        print("Для полного анализа установите:")
        print("  config.USE_SAMPLING = False")
        print("И перезапустите скрипт")

if __name__ == "__main__":
    main()
