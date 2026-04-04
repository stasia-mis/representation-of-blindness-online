# ============================================================
# ПАРСЕР VK: сбор постов из ЛИЧНЫХ БЛОГОВ незрячих пользователей
# 
# Логика:
#   1. Для каждой группы из списка GROUPS получаем её подписчиков.
#   2. Для каждого подписчика загружаем посты за указанный год (YEAR). Год разработки март 2026ого. 
#   3. Оставляем только посты, содержащие ключевые слова из KEYWORDS.
#   4. Сохраняем результат в CSV-файл.
#
# КАК ИСПОЛЬЗОВАТЬ:
#   1. Установите библиотеки: pip install vk_api pandas
#   2. Получите VK токен
#   3. Скопируйте токен и вставьте его ниже вместо "ваш_токен_здесь"
#   4. Замените список GROUPS на свои ID групп (из ссылок https://vk.com/club123456)
#   5. При необходимости измените YEAR, KEYWORDS и другие параметры
#   6. Запустите: python step2_blogger.py
# ДЛЯ УСКОРЕНИЯ СБОРА (если у вас несколько VK токенов):
#   1. Задюпайте парсер изменив выходной файл и имя парсера
#   2. В каждой копии замените VK_TOKEN на свой уникальный токен.
#   3. Pазделите список GROUPS между копиями вручную.
#   4. Запустите все копии одновременно – они будут работать параллельно.
# ============================================================

import vk_api
import pandas as pd
import time
from datetime import datetime

# ============================================================
# НАСТРОЙКИ (ОБЯЗАТЕЛЬНО ЗАМЕНИТЕ ТОКЕН И СПИСКИ ГРУПП!)
# ============================================================

# -------- VK ТОКЕН (ЗАМЕНИТЕ!) --------
VK_TOKEN = "ваш_токен_здесь"   # <--- ВСТАВЬТЕ СВОЙ ТОКЕН

# -------- СПИСОК ГРУПП (ЗАМЕНИТЕ!) --------
# ID берётся из ссылки: https://vk.com/club123456 → 123456
GROUPS = [
    111, 1111111, 11111, 111111, 111111, 11111,
    # добавьте остальные ID...
]

# -------- КЛЮЧЕВЫЕ СЛОВА (можно менять) --------
KEYWORDS = [
    "незрячий", "слепой", "слабовидящий", "потерял зрение", "брайл",
    "шрифт брайля", "VoiceOver", "тифлотехника", "адаптация",
    "реабилитация", "самостоятельность", "мобильная грамотность",
    "мой опыт", "жизнь без зрения", "особый взгляд", "тифлопедагог"
]

# -------- ПАРАМЕТРЫ СБОРА (можно менять) --------
YEAR = 2026                      # Год постов
MAX_SUBSCRIBERS_PER_GROUP = 500  # Сколько подписчиков анализировать на группу
MAX_POSTS_PER_BLOGGER = 50       # Максимум постов на одного блогера

# -------- ВЫХОДНОЙ ФАЙЛ --------
OUTPUT_FILE = "vk_bloggers_corpus.csv"

# ============================================================
# ПРОВЕРКА ТОКЕНА (если не заменён – подсказка)
# ============================================================
if VK_TOKEN == "ваш_токен_здесь":
    print("\n" + "="*60)
    print("❌ ОШИБКА: Вы не заменили VK_TOKEN на свой токен!")
    print("="*60)
    print("👉 Как получить токен:")
    print("   1. Перейдите по ссылке: https://vkhost.github.io/")
    print("   2. Отметьте права: wall, groups, users, offline")
    print("   3. Нажмите «Получить токен»")
    print("   4. Скопируйте полученную длинную строку")
    print("   5. Вставьте её в переменную VK_TOKEN вместо «ваш_токен_здесь»")
    print("\nПример: VK_TOKEN = \"vk1.a.6P8CJcr-5fPZoTGO7lh3...\"\n")
    exit(1)

# ============================================================
# ФУНКЦИИ
# ============================================================

def auth_vk():
    vk_session = vk_api.VkApi(token=VK_TOKEN)
    return vk_session.get_api()

def is_relevant_post(text):
    text = text.lower()
    return any(kw in text for kw in KEYWORDS)

def get_subscribers(vk, group_id, max_count=500):
    subscribers = []
    offset = 0
    while offset < max_count:
        try:
            response = vk.groups.getMembers(group_id=group_id, count=1000, offset=offset)
            items = response.get("items", [])
            if not items:
                break
            subscribers.extend(items)
            offset += 1000
            time.sleep(0.3)
        except Exception as e:
            print(f"Ошибка получения подписчиков группы {group_id}: {e}")
            break
    return subscribers[:max_count]

def get_user_posts(vk, user_id, year, max_posts=50):
    posts = []
    offset = 0
    try:
        while len(posts) < max_posts:
            response = vk.wall.get(owner_id=user_id, count=100, offset=offset)
            items = response.get("items", [])
            if not items:
                break
            for post in items:
                post_date = datetime.fromtimestamp(post.get("date", 0))
                if post_date.year < year:
                    # посты дальше только старее – можно выйти из цикла
                    return posts
                if post_date.year != year:
                    continue
                text = post.get("text", "").strip()
                if text and is_relevant_post(text):
                    posts.append({
                        "user_id": user_id,
                        "post_id": post.get("id"),
                        "date": post_date.strftime("%Y-%m-%d"),
                        "text": text,
                        "url": f"https://vk.com/id{user_id}"   # прямая ссылка на страницу
                    })
            offset += 100
            time.sleep(0.3)
    except Exception as e:
        print(f"Ошибка получения постов пользователя {user_id}: {e}")
    return posts

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("\n" + "="*60)
    print("ПАРСЕР БЛОГЕРОВ (по подписчикам групп)")
    print("="*60)
    
    vk = auth_vk()
    all_bloggers_posts = []

    for group_id in GROUPS:
        print(f"\nСобираем подписчиков группы {group_id}...")
        subscribers = get_subscribers(vk, group_id, max_count=MAX_SUBSCRIBERS_PER_GROUP)
        print(f"Найдено подписчиков: {len(subscribers)}")

        for i, user_id in enumerate(subscribers, 1):
            posts = get_user_posts(vk, user_id, year=YEAR, max_posts=MAX_POSTS_PER_BLOGGER)
            if posts:
                all_bloggers_posts.extend(posts)
                print(f"[{i}/{len(subscribers)}] Пользователь {user_id} → {len(posts)} релевантных постов")

    if all_bloggers_posts:
        df = pd.DataFrame(all_bloggers_posts)
        df.drop_duplicates(subset=["user_id", "post_id"], inplace=True)
        df.sort_values("date", ascending=False, inplace=True)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("\n" + "="*60)
        print(f"✅ ГОТОВО! Всего постов: {len(df)}")
        print(f"   Файл: {OUTPUT_FILE}")
        print("="*60)
    else:
        print("\n❌ Нет релевантных блогеров или постов за указанный год.")

if __name__ == "__main__":
    main()
