# ============================================================
# ПАРСЕР VK: поиск ТЕМАТИЧЕСКИХ ГРУПП (НКО, библиотеки для слепых, некомерческих и информационных проектов а так же личных блогов)
# и сбор постов из этих групп.
# 
# ЧТО ДЕЛАЕТ:
#   1. Ищет группы по ключевым словам (например, "незрячие", "тифлопедагогика").
#   2. Фильтрует группы по наличию релевантных слов в названии/описании.
#   3. Собирает посты из найденных групп (до 100 на группу).
#   4. Сохраняет результат в CSV с колонками: group_id, post_id, date, text, likes, comments, url.
#
# КАК ИСПОЛЬЗОВАТЬ:
#   1. Установите библиотеки: pip install vk_api pandas
#   2. Получите VK токен. я испольовала токен своей страницы в вк
#   3. Скопируйте полученный токен и вставьте его ниже вместо "ваш_токен_здесь"
#   4. При необходимости измените список KEYWORDS_SEARCH или лимиты MAX_POSTS_PER_GROUP
#   5. Запустите: python step1_parse_VKv2.py
# ============================================================

import vk_api
import pandas as pd
import time
from datetime import datetime

# ============================================================
# НАСТРОЙКИ (ОБЯЗАТЕЛЬНО ЗАМЕНИТЕ ТОКЕН!)
# ============================================================

# 1. ВСТАВЬТЕ СВОЙ ТОКЕН (между кавычек)
VK_TOKEN = "ваш_токен_здесь"   # <--- ЗАМЕНИТЕ ЭТО

# 2. Ключевые слова для поиска групп (можно менять)
KEYWORDS_SEARCH = [
    "слабовидящие",
    "незрячие",
    "инвалиды по зрению",
    "тифлопедагогика",
    "собака поводырь",
    "шрифт брайля",
]

# 3. Лимиты (можно менять)
MAX_GROUPS_PER_QUERY = 200   # Сколько групп обработать на один поисковый запрос
MAX_POSTS_PER_GROUP = 100    # Сколько постов собрать из каждой группы

# 4. Выходной файл
OUTPUT_FILE = "vk_research_corpus.csv"

# ============================================================
# ПРОВЕРКА: если токен не заменён
# ============================================================
if VK_TOKEN == "ваш_токен_здесь":
    print("\n" + "="*60)
    print("❌ ОШИБКА: Вы не заменили VK_TOKEN на свой токен!")
    print("="*60)
    print("\nПример: VK_TOKEN = \"vk1.a.6P8CJcr-5fPZoTGO7lh3...\"\n")
    exit(1)

# ============================================================
# ФИЛЬТРЫ
# ============================================================

def is_relevant_group(group):
    """
    Проверяет, подходит ли группа.
    - Хорошие слова: незряч, слабовид, инвалид, зрение, тифло, брайл
    - Плохие слова: фильм, кино, аниме, игра, юмор, мем, прикол
    """
    text = (group.get("name", "") + " " + group.get("description", "")).lower()
    good_words = ["незряч", "слабовид", "инвалид", "зрение", "тифло", "брайл"]
    bad_words  = ["фильм", "кино", "аниме", "игра", "юмор", "мем", "прикол"]
    
    if any(b in text for b in bad_words):
        return False
    return any(g in text for g in good_words)

def is_relevant_post(text):
    """Проверяет, содержит ли пост хотя бы одно ключевое слово"""
    text = text.lower()
    keywords = ["зрение", "незряч", "слабовид", "инвалид", "поводыр", "брайл"]
    return any(k in text for k in keywords)

# ============================================================
# ФУНКЦИИ РАБОТЫ С VK API
# ============================================================

def auth_vk():
    """Авторизация по токену"""
    vk_session = vk_api.VkApi(token=VK_TOKEN)
    return vk_session.get_api()

def search_groups(vk, query):
    """
    Ищет группы по ключевому слову query, фильтрует через is_relevant_group.
    Возвращает список ID групп.
    """
    groups_found = []
    offset = 0
    print(f"\nПоиск: {query}")
    
    while offset < MAX_GROUPS_PER_QUERY:
        response = vk.groups.search(
            q=query,
            type="group",
            count=100,
            offset=offset,
            fields="description"
        )
        items = response.get("items", [])
        if not items:
            break
        
        for g in items:
            if is_relevant_group(g):
                groups_found.append(g["id"])
        
        offset += 100
        time.sleep(0.3)  # пауза, чтобы не заблокировали API
    
    print(f"  Найдено подходящих групп: {len(groups_found)}")
    return groups_found

def collect_posts(vk, group_id):
    """
    Собирает посты из стены группы.
    Возвращает список словарей с данными постов.
    """
    posts = []
    offset = 0
    
    while len(posts) < MAX_POSTS_PER_GROUP:
        try:
            response = vk.wall.get(owner_id=-group_id, count=100, offset=offset)
            items = response.get("items", [])
            if not items:
                break
            
            for post in items:
                text = post.get("text", "").strip()
                if not text or not is_relevant_post(text):
                    continue
                
                posts.append({
                    "group_id": group_id,
                    "post_id": post["id"],
                    "date": datetime.fromtimestamp(post["date"]).strftime("%Y-%m-%d"),
                    "text": text,
                    "likes": post.get("likes", {}).get("count", 0),
                    "comments": post.get("comments", {}).get("count", 0),
                    "url": f"https://vk.com/club{group_id}"   # прямая ссылка на группу
                })
            
            offset += 100
            time.sleep(0.3)
        except Exception as e:
            print(f"  Ошибка при сборе постов группы {group_id}: {e}")
            break
    
    return posts

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("\n" + "="*60)
    print("ПАРСЕР ПОИСКА ГРУПП И СБОРА ПОСТОВ")
    print("="*60)
    
    vk = auth_vk()
    all_groups = set()  # множество, чтобы избежать дубликатов
    
    # 1. Поиск групп по каждому ключевому слову
    for kw in KEYWORDS_SEARCH:
        groups = search_groups(vk, kw)
        all_groups.update(groups)
    
    print(f"\n✅ Всего уникальных групп после фильтрации: {len(all_groups)}")
    
    # 2. Сбор постов из каждой найденной группы
    all_posts = []
    for i, group_id in enumerate(all_groups, 1):
        print(f"[{i}/{len(all_groups)}] Сбор постов из группы {group_id}...")
        posts = collect_posts(vk, group_id)
        all_posts.extend(posts)
        print(f"    Собрано релевантных постов: {len(posts)}")
    
    # 3. Сохранение результатов
    if all_posts:
        df = pd.DataFrame(all_posts)
        df.drop_duplicates(subset=["group_id", "post_id"], inplace=True)
        df.sort_values("date", ascending=False, inplace=True)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        
        print("\n" + "="*60)
        print(f"✅ ГОТОВО!")
        print(f"   Всего постов: {len(df)}")
        print(f"   Сохранено в файл: {OUTPUT_FILE}")
        print("="*60)
    else:
        print("\n❌ Нет подходящих постов для сохранения.")

if __name__ == "__main__":
    main()
