# ============================================================
# ПАРСЕР VK: сбор постов из РУЧНЫХ СПИСКОВ (после фильтрации)
# 
# ВАЖНО: этот скрипт запускается ТОЛЬКО ПОСЛЕ того, как вы вручную
# отобрали нужные ID на основе результатов первых двух парсеров.
#
# У вас есть два отдельных списка:
#   1. GROUPS – ID групп (НКО, библиотеки, некоммерческие проекты), либо айди личных каналов блогеров
#   2. USERS – ID личных страниц блогеров (найденных в на этапе 2)
#
# В зависимости от того, что вы собираете:
#   - Если собираете НКО → установите OUTPUT_TYPE = "nko"
#   - Если собираете личные блоги → установите OUTPUT_TYPE = "blogs"
#
# Скрипт соберёт все посты за указанный период (без фильтрации по ключевым словам)
# и сохранит в файл nko.csv или blogs.csv.
#
# КАК ИСПОЛЬЗОВАТЬ:
#   1. Установите: pip install vk_api pandas
#   2. Получите VK токен
#   3. Вставьте токен в переменную VK_TOKEN
#   4. Заполните список GROUPS (для НКО или БЛОГОВ) или USERS (для блогов) своими ID
#   5. Установите OUTPUT_TYPE = "nko" или "blogs"
#   6. Запустите: python step3_vk_corpus.py
#
# ============================================================

import vk_api
import pandas as pd
import time
from datetime import datetime

# ============================================================
# НАСТРОЙКИ (ОБЯЗАТЕЛЬНО ЗАМЕНИТЕ ТОКЕН!)
# ============================================================

VK_TOKEN = "ваш_токен_здесь"   # <--- ВСТАВЬТЕ СВОЙ ТОКЕН

# -------- ВЫБОР ТИПА ВЫХОДНОГО ФАЙЛА --------
# "blogs" – для личных блогов (сохранится в blogs.csv)
# "nko"   – для групп НКО (сохранится в nko.csv)
OUTPUT_TYPE = "blogs"   # <--- МЕНЯЙТЕ ВРУЧНУЮ ПРИ НЕОБХОДИМОСТИ

# -------- СПИСКИ ID (ЗАМЕНИТЕ НА СВОИ) --------
# Список групп 
GROUPS = [
    111, 111, 111, 111, 11,
    # добавьте ID групп 
]

# Список пользователей (личные страницы блогеров, отобранные вручную)
USERS = [
    11, 11, 11, 11, 11,
    # добавьте ID пользователей (только если OUTPUT_TYPE = "blogs")
]

DATE_FROM = "2020-01-01"
DATE_TO   = "2026-12-31"

MAX_POSTS_GROUP = 2000   # максимум постов из одной группы
MAX_POSTS_USER  = 1000   # максимум постов от одного пользователя

# -------- ВЫХОДНОЙ ФАЙЛ (ОПРЕДЕЛЯЕТСЯ АВТОМАТИЧЕСКИ) --------
if OUTPUT_TYPE == "blogs":
    OUTPUT_FILE = "blogs.csv"
elif OUTPUT_TYPE == "nko":
    OUTPUT_FILE = "nko.csv"
else:
    raise ValueError("OUTPUT_TYPE должен быть 'blogs' или 'nko'")

# ============================================================
# ПРОВЕРКА ТОКЕНА
# ============================================================
if VK_TOKEN == "ваш_токен_здесь":
    print("\n❌ ОШИБКА: Замените VK_TOKEN на свой токен!")
    exit(1)

# ============================================================
# ФУНКЦИИ (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================

def get_ts(date):
    return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

def auth_vk():
    vk_session = vk_api.VkApi(token=VK_TOKEN)
    return vk_session.get_api()

def collect_wall(vk, owner_id, is_group, max_posts, ts_from, ts_to):
    posts = []
    offset = 0
    batch = 100
    real_id = -owner_id if is_group else owner_id

    while len(posts) < max_posts:
        try:
            res = vk.wall.get(owner_id=real_id, count=batch, offset=offset)
            items = res.get("items", [])
            if not items:
                break
            for post in items:
                date_ts = post.get("date", 0)
                if date_ts < ts_from:
                    return posts
                if date_ts > ts_to:
                    continue
                text = post.get("text", "")
                if not text:
                    continue
                posts.append({
                    "source_type": "group" if is_group else "user",
                    "owner_id": owner_id,
                    "post_id": post.get("id"),
                    "date": datetime.fromtimestamp(date_ts).strftime("%Y-%m-%d"),
                    "text": text,
                    "likes": post.get("likes", {}).get("count", 0),
                    "comments": post.get("comments", {}).get("count", 0),
                    "reposts": post.get("reposts", {}).get("count", 0),
                    "views": post.get("views", {}).get("count", 0),
                    "url": f"https://vk.com/club{owner_id}" if is_group else f"https://vk.com/id{owner_id}"
                })
            offset += batch
            time.sleep(0.4)
        except Exception as e:
            print(f"Ошибка {owner_id}: {e}")
            break
    return posts

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    vk = auth_vk()
    all_data = []

    ts_from = get_ts(DATE_FROM)
    ts_to   = get_ts(DATE_TO)

    # ГРУППЫ
    for g in GROUPS:
        print(f"\nГруппа {g}")
        data = collect_wall(vk, g, True, MAX_POSTS_GROUP, ts_from, ts_to)
        all_data.extend(data)

    # ПОЛЬЗОВАТЕЛИ
    for u in USERS:
        print(f"\nПользователь {u}")
        data = collect_wall(vk, u, False, MAX_POSTS_USER, ts_from, ts_to)
        all_data.extend(data)

    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=["owner_id", "post_id"], inplace=True)
    df.sort_values("date", ascending=False, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nГотово: {len(df)} строк -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
