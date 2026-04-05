"""
СКРИПТ ДЛЯ РУЧНОЙ ВАЛИДАЦИИ WEAK LABELS

Этот скрипт НЕОБХОДИМ ДЛЯ ТОГО ЧТОБЫ:
1. Создать выборку для ручной проверки
2. Экспортировать в Excel для удобной разметки
3. Сравнить ручные метки с автоматическими
4. Вычислить метрики качества
5. Рекомендовать оптимальный threshold
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import random

# ============================================================================
# ШАГ 1: СОЗДАНИЕ ВЫБОРКИ ДЛЯ РУЧНОЙ ПРОВЕРКИ
# ============================================================================

def create_validation_sample(nko_path='output/nko_sample_predictions.csv',
                             blogs_path='output/blogs_sample_predictions.csv',
                             n_per_source=50):
    """
    Создать стратифицированную выборку для ручной проверки
    
    Parameters:
    - n_per_source: сколько текстов взять из каждого источника (НКО/блоги)
    """
    print("="*60)
    print("СОЗДАНИЕ ВЫБОРКИ ДЛЯ РУЧНОЙ ВАЛИДАЦИИ")
    print("="*60)
    
    # Загрузить данные
    nko = pd.read_csv(nko_path)
    blogs = pd.read_csv(blogs_path)
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    
    # Стратифицированная выборка
    # Берем равное количество постов с каждой меткой и без нее
    def stratified_sample(df, n):
        samples = []
        
        # Для каждой категории взять примеры с меткой и без
        for cat in categories:
            # С меткой
            with_label = df[df[f'{cat}_label'] == 1].sample(
                n=min(n // 8, len(df[df[f'{cat}_label'] == 1])),
                random_state=42
            )
            samples.append(with_label)
            
            # Без метки
            without_label = df[df[f'{cat}_label'] == 0].sample(
                n=min(n // 8, len(df[df[f'{cat}_label'] == 0])),
                random_state=42
            )
            samples.append(without_label)
        
        # Объединить и перемешать
        result = pd.concat(samples).drop_duplicates()
        
        # Если не хватает, добавить случайные
        if len(result) < n:
            remaining = df[~df.index.isin(result.index)].sample(
                n=n - len(result),
                random_state=42
            )
            result = pd.concat([result, remaining])
        
        return result.sample(frac=1, random_state=42).head(n)  # Перемешать
    
    # Создать выборки
    nko_sample = stratified_sample(nko, n_per_source)
    blogs_sample = stratified_sample(blogs, n_per_source)
    
    # Объединить
    validation_sample = pd.concat([nko_sample, blogs_sample], ignore_index=True)
    
    # Подготовить для экспорта
    export_df = pd.DataFrame({
        'id': validation_sample.index,
        'source': validation_sample['source'],
        'text': validation_sample['text'],
        
        # Автоматические метки
        'auto_victim': validation_sample['victim_label'],
        'auto_supercrip': validation_sample['supercrip_label'],
        'auto_agency': validation_sample['agency_label'],
        'auto_first_person': validation_sample['first_person_label'],
        
        # Scores для контекста
        'victim_score': validation_sample['victim_score'],
        'supercrip_score': validation_sample['supercrip_score'],
        'agency_score': validation_sample['agency_score'],
        'first_person_score': validation_sample['first_person_score'],
        
        # Пустые столбцы для ручной разметки
        'manual_victim': '',
        'manual_supercrip': '',
        'manual_agency': '',
        'manual_first_person': '',
        
        # Столбец для комментариев
        'comments': ''
    })
    
    # Сохранить
    export_df.to_excel('validation_sample.xlsx', index=False)
    export_df.to_csv('validation_sample.csv', index=False)
    
    print(f"\n✅ Создана выборка: {len(export_df)} постов")
    print(f"   НКО: {len(nko_sample)} постов")
    print(f"   Блоги: {len(blogs_sample)} постов")
    print(f"\nФайлы сохранены:")
    print("   - validation_sample.xlsx (для Excel)")
    print("   - validation_sample.csv (для других программ)")
    
    # Показать распределение
    print("\nРаспределение автоматических меток:")
    for cat in categories:
        auto_col = f'auto_{cat}'
        count = export_df[auto_col].sum()
        pct = (count / len(export_df)) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")
    
    return export_df

# ============================================================================
# ШАГ 2: ИНСТРУКЦИИ ДЛЯ РУЧНОЙ РАЗМЕТКИ
# ============================================================================

def print_manual_labeling_instructions():
    """Вывести инструкции для ручной разметки"""
    
    instructions = """
╔════════════════════════════════════════════════════════════════════╗
║         ИНСТРУКЦИЯ ПО РУЧНОЙ РАЗМЕТКЕ VALIDATION_SAMPLE           ║
╚════════════════════════════════════════════════════════════════════╝

1. ОТКРОЙ ФАЙЛ:
   validation_sample.xlsx в Excel

2. ДЛЯ КАЖДОГО ПОСТА (строки) ЗАПОЛНИ СТОЛБЦЫ:
   - manual_victim: 1 или 0
   - manual_supercrip: 1 или 0
   - manual_agency: 1 или 0
   - manual_first_person: 1 или 0

3. КАК ОПРЕДЕЛЯТЬ КАТЕГОРИИ:

   📌 VICTIM (Жертва):
   ✅ Ставь 1 если:
      - Упоминается страдание, беспомощность, нужда
      - Акцент на ограничениях, проблемах
      - Призывы помочь, пожертвовать
      - Человек представлен как объект благотворительности
   ❌ Ставь 0 если:
      - Просто упоминание инвалидности без акцента на страдание
      - Нейтральное описание

   Примеры:
   1: "Помогите тем, кто страдает от слепоты и нуждается в поддержке"
   0: "Организация для незрячих людей"

   📌 SUPERCRIP (Суперкалека):
   ✅ Ставь 1 если:
      - Героизация, акцент на преодолении
      - "Несмотря на диагноз", "вопреки недугу"
      - Обычные действия представлены как подвиг
   ❌ Ставь 0 если:
      - Обычные достижения без героизации
      - Нейтральное упоминание успехов

   Примеры:
   1: "Несмотря на слепоту, он смог окончить университет!"
   0: "Я окончил университет"

   📌 AGENCY (Агентность):
   ✅ Ставь 1 если:
      - Самостоятельность, активность
      - Работа, учеба, создание чего-то
      - Принятие решений, выбор
      - Независимые действия
   ❌ Ставь 0 если:
      - Только получение помощи

   Примеры:
   1: "Я работаю программистом и создаю приложения"
   0: "Мне помогают пользоваться компьютером"

   📌 FIRST_PERSON (Я-высказывание):
   ✅ Ставь 1 если:
      - Есть "я", "мы", "мой", "наш"
      - Личная позиция, опыт
   ❌ Ставь 0 если:
      - Только третье лицо
      - Безличные конструкции

   Примеры:
   1: "Я считаю, что наше общество должно измениться"
   0: "Слепые люди сталкиваются с проблемами"

4. ВАЖНО:
   - Один пост может иметь НЕСКОЛЬКО меток (например, 1,0,1,1)
   - Или НИ ОДНОЙ метки (0,0,0,0)
   - Оценивайте ВЕСЬ текст, не отдельные слова
   - В столбце "comments" можете писать пояснения

5. ПОСЛЕ ЗАПОЛНЕНИЯ:
   - Сохраните файл
   - Запустите скрипт для проверки качества (см. ниже)

"""
    
    print(instructions)

# ============================================================================
# ШАГ 3: СРАВНЕНИЕ РУЧНЫХ И АВТОМАТИЧЕСКИХ МЕТОК
# ============================================================================

def evaluate_weak_labels(manual_file='validation_sample.xlsx'):
    """
    Сравнить ручные метки с автоматическими и вычислить метрики
    """
    print("\n" + "="*60)
    print("ОЦЕНКА КАЧЕСТВА WEAK LABELS")
    print("="*60)
    
    # Загрузить файл с ручной разметкой
    df = pd.read_excel(manual_file)
    
    # Проверить что разметка заполнена
    manual_cols = ['manual_victim', 'manual_supercrip', 'manual_agency', 'manual_first_person']
    
    if df[manual_cols].isna().all().all():
        print("\n⚠️  ОШИБКА: Ручная разметка не заполнена!")
        print("Заполните столбцы manual_* в файле validation_sample.xlsx")
        return None
    
    # Проверить заполненность
    filled = (~df[manual_cols].isna()).all(axis=1).sum()
    print(f"\nЗаполнено разметок: {filled} / {len(df)}")
    
    if filled < 50:
        print("⚠️  Слишком мало размеченных постов. Рекомендуется минимум 50.")
        response = input("Продолжить с имеющимися? (y/n): ")
        if response.lower() != 'y':
            return None
    
    # Оставить только заполненные строки
    df = df[~df[manual_cols].isna().any(axis=1)]
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    results = []
    
    print("\n" + "="*60)
    print("МЕТРИКИ ПО КАТЕГОРИЯМ")
    print("="*60)
    
    for cat in categories:
        auto_col = f'auto_{cat}'
        manual_col = f'manual_{cat}'
        
        y_true = df[manual_col].values
        y_pred = df[auto_col].values
        
        # Метрики
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'category': cat,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
        
        print(f"\n{cat.upper()}:")
        print(f"  Precision: {precision:.3f} (из предсказанных 1, сколько правильных)")
        print(f"  Recall:    {recall:.3f} (из реальных 1, сколько нашли)")
        print(f"  F1-score:  {f1:.3f} (общее качество)")
        print(f"  Confusion matrix:")
        print(f"    True Positives:  {tp} (правильно нашли)")
        print(f"    False Positives: {fp} (лишние, ошибочно нашли)")
        print(f"    True Negatives:  {tn} (правильно не нашли)")
        print(f"    False Negatives: {fn} (пропустили)")
    
    # Общая сводка
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("ОБЩАЯ СВОДКА")
    print("="*60)
    print(results_df[['category', 'precision', 'recall', 'f1']].to_string(index=False))
    
    avg_precision = results_df['precision'].mean()
    avg_recall = results_df['recall'].mean()
    avg_f1 = results_df['f1'].mean()
    
    print(f"\nСредние метрики:")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall:    {avg_recall:.3f}")
    print(f"  F1:        {avg_f1:.3f}")
    
    # Оценка качества
    print("\n" + "="*60)
    print("ОЦЕНКА КАЧЕСТВА")
    print("="*60)
    
    if avg_f1 >= 0.70:
        print("✅ ОТЛИЧНО! Качество weak labels высокое.")
    elif avg_f1 >= 0.60:
        print("✅ ХОРОШО! Качество приемлемое для анализа.")
    elif avg_f1 >= 0.50:
        print("⚠️  УДОВЛЕТВОРИТЕЛЬНО. Рекомендуется улучшить словари маркеров.")
    else:
        print("❌ НИЗКОЕ качество. Необходимо пересмотреть словари и threshold.")
    
    # Сохранить результаты
    results_df.to_csv('weak_labels_quality.csv', index=False)
    print("\nРезультаты сохранены в: weak_labels_quality.csv")
    
    return results_df

# ============================================================================
# ШАГ 4: РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ
# ============================================================================

def analyze_errors_and_recommend(manual_file='validation_sample.xlsx'):
    """
    Анализ ошибок и рекомендации по улучшению
    """
    print("\n" + "="*60)
    print("АНАЛИЗ ОШИБОК И РЕКОМЕНДАЦИИ")
    print("="*60)
    
    df = pd.read_excel(manual_file)
    manual_cols = ['manual_victim', 'manual_supercrip', 'manual_agency', 'manual_first_person']
    df = df[~df[manual_cols].isna().any(axis=1)]
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    
    for cat in categories:
        auto_col = f'auto_{cat}'
        manual_col = f'manual_{cat}'
        score_col = f'{cat}_score'
        
        print(f"\n{'='*60}")
        print(f"{cat.upper()}")
        print(f"{'='*60}")
        
        # False Positives (автомат сказал 1, вручную 0)
        fp = df[(df[auto_col] == 1) & (df[manual_col] == 0)]
        
        if len(fp) > 0:
            print(f"\n❌ FALSE POSITIVES ({len(fp)}): Автомат нашел, но это неправильно")
            print("Примеры:")
            for idx, row in fp.head(3).iterrows():
                print(f"\n  Score: {row[score_col]:.2f}")
                print(f"  Text: {row['text'][:150]}...")
            
            print(f"\nРекомендация:")
            print(f"  → Проверьте словарь маркеров для '{cat}'")
            print(f"  → Возможно, некоторые слова слишком общие")
            print(f"  → Рассмотрите повышение threshold с 1.0 до 1.5 или 2.0")
        
        # False Negatives (автомат сказал 0, вручную 1)
        fn = df[(df[auto_col] == 0) & (df[manual_col] == 1)]
        
        if len(fn) > 0:
            print(f"\n❌ FALSE NEGATIVES ({len(fn)}): Автомат пропустил")
            print("Примеры:")
            for idx, row in fn.head(3).iterrows():
                print(f"\n  Score: {row[score_col]:.2f}")
                print(f"  Text: {row['text'][:150]}...")
            
            print(f"\nРекомендация:")
            print(f"  → Добавьте дополнительные маркеры в словарь для '{cat}'")
            print(f"  → Рассмотрите снижение threshold с 1.0 до 0.5")
            print(f"  → Обратите внимание на синонимы и морфологические варианты")

# ============================================================================
# ШАГ 5: ОПТИМИЗАЦИЯ THRESHOLD
# ============================================================================

def optimize_threshold(manual_file='validation_sample.xlsx'):
    """
    Найти оптимальный threshold для каждой категории
    """
    print("\n" + "="*60)
    print("ОПТИМИЗАЦИЯ THRESHOLD")
    print("="*60)
    
    df = pd.read_excel(manual_file)
    manual_cols = ['manual_victim', 'manual_supercrip', 'manual_agency', 'manual_first_person']
    df = df[~df[manual_cols].isna().any(axis=1)]
    
    categories = ['victim', 'supercrip', 'agency', 'first_person']
    recommendations = []
    
    for cat in categories:
        manual_col = f'manual_{cat}'
        score_col = f'{cat}_score'
        
        y_true = df[manual_col].values
        scores = df[score_col].values
        
        # Попробовать разные threshold
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        results = []
        
        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        results_df = pd.DataFrame(results)
        best = results_df.loc[results_df['f1'].idxmax()]
        
        print(f"\n{cat.upper()}:")
        print(results_df.to_string(index=False))
        print(f"\n  ✅ Лучший threshold: {best['threshold']} (F1 = {best['f1']:.3f})")
        
        recommendations.append({
            'category': cat,
            'current_threshold': 1.0,
            'optimal_threshold': best['threshold'],
            'current_f1': results_df[results_df['threshold'] == 1.0]['f1'].values[0],
            'optimal_f1': best['f1']
        })
    
    recommendations_df = pd.DataFrame(recommendations)
    
    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИИ ПО THRESHOLD")
    print("="*60)
    print(recommendations_df.to_string(index=False))
    
    # Сохранить
    recommendations_df.to_csv('threshold_recommendations.csv', index=False)
    print("\nСохранено в: threshold_recommendations.csv")
    
    return recommendations_df

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основной workflow валидации"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           ВАЛИДАЦИЯ WEAK LABELS - ГЛАВНОЕ МЕНЮ                ║
╚═══════════════════════════════════════════════════════════════╝

Выберите действие:

1. Создать выборку для ручной проверки (validation_sample.xlsx)
2. Показать инструкции по ручной разметке
3. Оценить качество weak labels (после ручной разметки)
4. Проанализировать ошибки и получить рекомендации
5. Оптимизировать threshold
6. Выполнить всё последовательно

0. Выход
""")
    
    choice = input("Ваш выбор (1-6): ").strip()
    
    if choice == '1':
        create_validation_sample(n_per_source=50)
        print("\n✅ Теперь откройте validation_sample.xlsx и заполните столбцы manual_*")
        
    elif choice == '2':
        print_manual_labeling_instructions()
        
    elif choice == '3':
        evaluate_weak_labels()
        
    elif choice == '4':
        analyze_errors_and_recommend()
        
    elif choice == '5':
        optimize_threshold()
        
    elif choice == '6':
        # Последовательное выполнение
        print("\n[1/5] Создание выборки...")
        create_validation_sample(n_per_source=50)
        
        print("\n[2/5] Инструкции...")
        print_manual_labeling_instructions()
        
        input("\nЗаполните validation_sample.xlsx и нажмите Enter для продолжения...")
        
        print("\n[3/5] Оценка качества...")
        evaluate_weak_labels()
        
        print("\n[4/5] Анализ ошибок...")
        analyze_errors_and_recommend()
        
        print("\n[5/5] Оптимизация threshold...")
        optimize_threshold()
        
        print("\n✅ ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ!")
    
    elif choice == '0':
        print("Выход")
    else:
        print("Неверный выбор")

if __name__ == "__main__":
    main()
