"""
МОДУЛЬ ВИЗУАЛИЗАЦИИ ДЛЯ АНАЛИЗА
Работает с результатами из modelv2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Поддержка русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_data():
    """
    Загрузить данные из output/
    """
    print("Загрузка данных...")
    
    data = {}
    
    # Попробовать загрузить разные файлы
    files_to_try = {
        'nko': [
            'output/nko_sample_predictions.csv',
            'output/nko_analyzed.csv',
            'nko_sample_predictions.csv'
        ],
        'blogs': [
            'output/blogs_sample_predictions.csv',
            'output/blogs_analyzed.csv',
            'blogs_sample_predictions.csv'
        ],
        'stats': [
            'output/statistical_comparison.csv',
            'statistical_comparison.csv'
        ]
    }
    
    for key, paths in files_to_try.items():
        loaded = False
        for path in paths:
            try:
                df = pd.read_csv(path)
                data[key] = df
                print(f"✅ Загружено: {path} ({len(df)} строк)")
                loaded = True
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"⚠️  Ошибка при загрузке {path}: {e}")
        
        if not loaded and key != 'stats':  # stats опциональный
            print(f"❌ Не удалось загрузить данные для {key}")
            print(f"   Проверьте что файлы существуют в папке output/")
    
    if 'nko' not in data or 'blogs' not in data:
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Нет данных для визуализации")
        print("Запустите сначала: python improved_analysis_BIG_DATA.py")
        return None
    
    return data

# ============================================================================
# ВИЗУАЛИЗАЦИИ
# ============================================================================

def plot_category_distribution(data, save_path='output/'):
    """
    График распределения категорий: НКО vs Блоги
    """
    print("\n[1/7] Создание графика распределения категорий...")
    
    nko = data['nko']
    blogs = data['blogs']
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    
    # Вычислить проценты
    nko_pct = []
    blogs_pct = []
    
    for cat in categories:
        label_col = f'{cat}_label'
        
        if label_col in nko.columns and label_col in blogs.columns:
            nko_pct.append((nko[label_col].sum() / len(nko)) * 100)
            blogs_pct.append((blogs[label_col].sum() / len(blogs)) * 100)
        else:
            print(f"⚠️  Столбец {label_col} не найден, используем scores")
            # Fallback на scores
            score_col = f'{cat}_score'
            nko_pct.append((nko[score_col] >= 1.0).sum() / len(nko) * 100)
            blogs_pct.append((blogs[score_col] >= 1.0).sum() / len(blogs) * 100)
    
    # График
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nko_pct, width, label='НКО', color='#e74c3c')
    bars2 = ax.bar(x + width/2, blogs_pct, width, label='Блоги', color='#3498db')
    
    # Подписи
    ax.set_xlabel('Категории дискурса', fontsize=14, fontweight='bold')
    ax.set_ylabel('Процент постов (%)', fontsize=14, fontweight='bold')
    ax.set_title('Распределение дискурсивных категорий: НКО vs Блоги', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Жертва\n(victim)', 'Суперкалека\n(supercrip)', 
                        'Агентность\n(agency)', 'Я-высказывание\n(first_person)'])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Значения на барах
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}category_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {save_path}category_distribution.png")
    plt.close()

def plot_scores_comparison(data, save_path='output/'):
    """
    Boxplot сравнение scores НКО vs Блоги
    """
    print("\n[2/7] Создание boxplot сравнения scores...")
    
    nko = data['nko']
    blogs = data['blogs']
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, cat in enumerate(categories):
        score_col = f'{cat}_score'
        
        if score_col not in nko.columns:
            print(f"⚠️  {score_col} не найден")
            continue
        
        # Подготовить данные
        plot_data = pd.DataFrame({
            'Score': list(nko[score_col]) + list(blogs[score_col]),
            'Источник': ['НКО']*len(nko) + ['Блоги']*len(blogs)
        })
        
        # Boxplot
        sns.boxplot(data=plot_data, x='Источник', y='Score', ax=axes[idx],
                   palette=['#e74c3c', '#3498db'])
        
        axes[idx].set_title(f'{cat.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Score (частота маркеров)', fontsize=11)
        axes[idx].set_xlabel('')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Сравнение scores по категориям: НКО vs Блоги', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{save_path}scores_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {save_path}scores_comparison.png")
    plt.close()

def plot_correlation_matrix(data, save_path='output/'):
    """
    Матрица корреляций между категориями
    """
    print("\n[3/7] Создание матрицы корреляций...")
    
    # Объединить данные
    combined = pd.concat([data['nko'], data['blogs']], ignore_index=True)
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    score_cols = [f'{cat}_score' for cat in categories]
    
    # Проверить наличие столбцов
    available_cols = [col for col in score_cols if col in combined.columns]
    
    if len(available_cols) < 2:
        print("⚠️  Недостаточно данных для корреляционной матрицы")
        return
    
    # Вычислить корреляции
    corr_matrix = combined[available_cols].corr()
    
    # Переименовать для красоты
    rename_dict = {
        'victim_score': 'Жертва',
        'supercrip_score': 'Суперкалека',
        'agency_score': 'Агентность',
        'first_person_score': 'Я-высказывание'
    }
    corr_matrix = corr_matrix.rename(columns=rename_dict, index=rename_dict)
    
    # График
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, vmin=-1, vmax=1, square=True, ax=ax,
               cbar_kws={'label': 'Корреляция Пирсона'})
    
    ax.set_title('Корреляции между дискурсивными категориями', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {save_path}correlation_matrix.png")
    plt.close()

def plot_statistical_tests(data, save_path='output/'):
    """
    График p-values статистических тестов
    """
    print("\n[4/7] Создание графика статистических тестов...")
    
    if 'stats' not in data or data['stats'] is None:
        print("⚠️  Нет файла statistical_comparison.csv, создаю из данных...")
        # Создать вручную
        from scipy.stats import mannwhitneyu
        
        nko = data['nko']
        blogs = data['blogs']
        categories = ['victim', 'supercrip', 'agency', 'first_person']
        
        stats_data = []
        for cat in categories:
            score_col = f'{cat}_score'
            if score_col in nko.columns and score_col in blogs.columns:
                _, p_val = mannwhitneyu(nko[score_col], blogs[score_col])
                stats_data.append({
                    'category': cat,
                    'p_value': p_val
                })
        
        stats_df = pd.DataFrame(stats_data)
    else:
        stats_df = data['stats']
    
    if 'p_value' not in stats_df.columns:
        print("⚠️  Нет столбца p_value в статистике")
        return
    
    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = stats_df['category'].values
    p_values = stats_df['p_value'].values
    
    # Log scale для лучшей видимости
    log_p = -np.log10(p_values)
    
    bars = ax.bar(categories, log_p, color=['red' if p < 0.05 else 'gray' 
                                             for p in p_values])
    
    # Линия значимости
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', 
              label='p = 0.05', linewidth=2)
    ax.axhline(y=-np.log10(0.01), color='red', linestyle='--', 
              label='p = 0.01', linewidth=2)
    
    ax.set_xlabel('Категории', fontsize=14, fontweight='bold')
    ax.set_ylabel('-log₁₀(p-value)', fontsize=14, fontweight='bold')
    ax.set_title('Статистическая значимость различий НКО vs Блоги\n(Mann-Whitney U test)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(['Жертва', 'Суперкалека', 'Агентность', 'Я-высказывание'])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Подписи p-values
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        if p < 0.001:
            label = 'p<0.001***'
        elif p < 0.01:
            label = f'p={p:.3f}**'
        elif p < 0.05:
            label = f'p={p:.3f}*'
        else:
            label = f'p={p:.3f}'
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}statistical_tests.png', dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {save_path}statistical_tests.png")
    plt.close()

def plot_text_length_distribution(data, save_path='output/'):
    """
    Распределение длины текстов
    """
    print("\n[5/7] Создание графика распределения длины текстов...")
    
    nko = data['nko']
    blogs = data['blogs']
    
    # Вычислить длину если нет
    if 'text' in nko.columns:
        nko['text_length'] = nko['text'].str.len()
        blogs['text_length'] = blogs['text'].str.len()
    else:
        print("⚠️  Столбец 'text' не найден, пропускаем")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(nko['text_length'], bins=50, alpha=0.6, label='НКО', color='#e74c3c')
    axes[0].hist(blogs['text_length'], bins=50, alpha=0.6, label='Блоги', color='#3498db')
    axes[0].set_xlabel('Длина текста (символы)', fontsize=12)
    axes[0].set_ylabel('Частота', fontsize=12)
    axes[0].set_title('Распределение длины текстов', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Boxplot
    plot_data = pd.DataFrame({
        'Длина': list(nko['text_length']) + list(blogs['text_length']),
        'Источник': ['НКО']*len(nko) + ['Блоги']*len(blogs)
    })
    
    sns.boxplot(data=plot_data, x='Источник', y='Длина', ax=axes[1],
               palette=['#e74c3c', '#3498db'])
    axes[1].set_title('Сравнение длины текстов', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Длина (символы)', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}text_length_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {save_path}text_length_distribution.png")
    plt.close()

def plot_multi_label_distribution(data, save_path='output/'):
    """
    Распределение количества меток на пост
    """
    print("\n[6/7] Создание графика multi-label распределения...")
    
    nko = data['nko']
    blogs = data['blogs']
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    label_cols = [f'{cat}_label' for cat in categories]
    
    # Проверить наличие
    available = [col for col in label_cols if col in nko.columns]
    
    if len(available) == 0:
        print("⚠️  Нет столбцов с метками")
        return
    
    # Подсчитать количество меток
    nko['n_labels'] = nko[available].sum(axis=1)
    blogs['n_labels'] = blogs[available].sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Распределение
    nko_counts = nko['n_labels'].value_counts().sort_index()
    blogs_counts = blogs['n_labels'].value_counts().sort_index()
    
    x = np.arange(0, max(nko_counts.index.max(), blogs_counts.index.max()) + 1)
    
    nko_values = [nko_counts.get(i, 0) for i in x]
    blogs_values = [blogs_counts.get(i, 0) for i in x]
    
    width = 0.35
    ax.bar(x - width/2, nko_values, width, label='НКО', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, blogs_values, width, label='Блоги', color='#3498db', alpha=0.7)
    
    ax.set_xlabel('Количество категорий на пост', fontsize=14, fontweight='bold')
    ax.set_ylabel('Частота', fontsize=14, fontweight='bold')
    ax.set_title('Распределение количества дискурсивных категорий\n(Multi-label характеристика)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}multi_label_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {save_path}multi_label_distribution.png")
    plt.close()

def plot_category_cooccurrence(data, save_path='output/'):
    """
    Матрица совместной встречаемости категорий
    """
    print("\n[7/7] Создание матрицы совместной встречаемости...")
    
    combined = pd.concat([data['nko'], data['blogs']], ignore_index=True)
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    label_cols = [f'{cat}_label' for cat in categories]
    
    available = [col for col in label_cols if col in combined.columns]
    
    if len(available) < 2:
        print("⚠️  Недостаточно данных для матрицы совместной встречаемости")
        return
    
    # Вычислить совместную встречаемость
    cooc_matrix = np.zeros((len(available), len(available)))
    
    for i, col1 in enumerate(available):
        for j, col2 in enumerate(available):
            # Процент постов где обе категории присутствуют
            both = ((combined[col1] == 1) & (combined[col2] == 1)).sum()
            cooc_matrix[i, j] = (both / len(combined)) * 100
    
    # Переименовать
    labels = ['Жертва', 'Суперкалека', 'Агентность', 'Я-высказывание'][:len(available)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cooc_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
               square=True, ax=ax, xticklabels=labels, yticklabels=labels,
               cbar_kws={'label': '% постов'})
    
    ax.set_title('Совместная встречаемость дискурсивных категорий\n(% от всех постов)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}category_cooccurrence.png', dpi=300, bbox_inches='tight')
    print(f"✅ Сохранено: {save_path}category_cooccurrence.png")
    plt.close()

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Создать все визуализации
    """
    print("="*70)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("="*70)
    
    # Создать output если нет
    Path('output').mkdir(exist_ok=True)
    
    # Загрузить данные
    data = load_data()
    
    if data is None:
        return
    
    # Создать визуализации
    try:
        plot_category_distribution(data)
    except Exception as e:
        print(f"❌ Ошибка в category_distribution: {e}")
    
    try:
        plot_scores_comparison(data)
    except Exception as e:
        print(f"❌ Ошибка в scores_comparison: {e}")
    
    try:
        plot_correlation_matrix(data)
    except Exception as e:
        print(f"❌ Ошибка в correlation_matrix: {e}")
    
    try:
        plot_statistical_tests(data)
    except Exception as e:
        print(f"❌ Ошибка в statistical_tests: {e}")
    
    try:
        plot_text_length_distribution(data)
    except Exception as e:
        print(f"❌ Ошибка в text_length_distribution: {e}")
    
    try:
        plot_multi_label_distribution(data)
    except Exception as e:
        print(f"❌ Ошибка в multi_label_distribution: {e}")
    
    try:
        plot_category_cooccurrence(data)
    except Exception as e:
        print(f"❌ Ошибка в category_cooccurrence: {e}")
    
    print("\n" + "="*70)
    print("✅ ВИЗУАЛИЗАЦИИ ЗАВЕРШЕНЫ")
    print("="*70)
    print("\nГотовые графики находятся в папке output/:")
    print("  - category_distribution.png")
    print("  - scores_comparison.png")
    print("  - correlation_matrix.png")
    print("  - statistical_tests.png")
    print("  - text_length_distribution.png")
    print("  - multi_label_distribution.png")
    print("  - category_cooccurrence.png")

if __name__ == "__main__":
    main()
