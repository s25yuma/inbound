# opinions.py -- 多数派／少数派の意見の集計
# 多数派 = 回答数の多い上位N件 / 少数派 = 回答数がK人以下の意見（多数派に入らないもの）
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

TOPICS = {  # key: (表示名, 種類)  種類: text=自由記述 / list=複数地点 / cat=単一選択 / multi=複数選択
    "f3_text": ("訪日前に楽しみにしていた観光・行動（自由記述）", "text"),
    "f2_text": ("最も満足した飲食（自由記述）", "text"),
    "f1_text": ("最も満足した購入商品（自由記述）", "text"),
    "spots": ("山形県内の訪問スポット", "list"),
    "co_prefs": ("山形県とあわせて訪問した都道府県", "list"),
    "product": ("最も満足した購入商品（17区分）", "cat"),
    "food": ("最も満足した飲食（13区分）", "cat"),
    "f1_reason": ("購入商品に満足した理由", "multi"),
    "f2_reason": ("飲食に満足した理由", "multi"),
}

_PAREN = re.compile(r"[（(][^）)]*[）)]")
_SPLIT = re.compile(r"[、，,/／\n]+")

# 「山形関連の意見」の判定に使うキーワード（簡易判定。足りない地名・名物はここに追加する）
YAMAGATA_WORDS = [
    "山形",
    "米沢",
    "銀山",
    "天童",
    "山寺",
    "立石寺",
    "尾花沢",
    "長井",
    "羽黒",
    "出羽",
    "朝日鷹",
    "おしん",
    "鷹山",
    "鶴岡",
    "酒田",
    "庄内",
    "蔵王温泉",
    "蔵王のロープウェイ",
    "蔵王ロープウェイ",
]


def _is_yamagata(text):
    t = unicodedata.normalize("NFKC", str(text))
    return any(w in t for w in YAMAGATA_WORDS)


def _norm(s, merge):
    """表記ゆれを統合するキー。merge=True なら括弧内の補足（例: 牛タン（仙台））を無視する。"""
    t = unicodedata.normalize("NFKC", str(s)).strip()
    if merge:
        t = _PAREN.sub("", t).strip() or t
    return re.sub(r"[\s・,、。!~〜\-]+", "", t).casefold()


def _display(s, merge):
    s = str(s).strip()
    return (_PAREN.sub("", s).strip() or s) if merge else s


def count_items(d, topic, multi, merge=True, split=True, pref_only=False):
    """意見ごとの回答数を数える。戻り値: (key->回答数, key->表示名, 回答者数)"""
    kind = TOPICS[topic][1]
    counts, forms = Counter(), defaultdict(Counter)

    def add(raw, parts=False):
        # parts=True: 1つの回答に複数の意見が入っている場合に区切り、同じ意見は1人1回だけ数える
        seen = {}
        for piece in (_SPLIT.split(str(raw)) if parts else [raw]):
            if pref_only and not _is_yamagata(piece):
                continue
            k = _norm(piece, merge)
            if k and k not in seen:
                seen[k] = _display(piece, merge)
        for k, disp in seen.items():
            counts[k] += 1
            forms[k][disp] += 1

    if kind == "text":
        vals = d[topic].dropna()
        for v in vals:
            add(v, parts=split)
        base = len(vals)
    elif kind == "list":
        base = int((d[topic].str.len() > 0).sum())
        for lst in d[topic]:
            for v in lst:
                add(v)
    elif kind == "cat":
        vals = d[topic].dropna()
        for v in vals:
            add(v)
        base = len(vals)
    else:  # multi: 選ばれなかった選択肢(0件)も少数派として残す
        pre, labels = multi[topic]
        cols = [f"{pre}_{i}" for i in range(1, len(labels) + 1)]
        base = int(d[cols].notna().any(axis=1).sum())
        for c, lab in zip(cols, labels):
            counts[lab] = int((d[c] == 1).sum())
            forms[lab][lab] = 1
    shown = {k: f.most_common(1)[0][0] for k, f in forms.items()}
    return counts, shown, base


def build_opinions(
    d,
    topic,
    multi,
    top=10,
    max_count=1,
    q="",
    merge=True,
    split=True,
    pref_only=False,
    limit=2000,
):
    label, kind = TOPICS[topic]
    counts, shown, base = count_items(
        d, topic, multi, merge, split, pref_only and kind == "text"
    )
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], shown[kv[0]]))
    major = ranked[:top]
    major_keys = {k for k, _ in major}
    # 少数派 = 多数派（上位N件）以外すべて。max_count>0 のときだけ「その人数以下」にさらに絞る
    minor_all = [
        (k, c)
        for k, c in ranked
        if k not in major_keys and (max_count == 0 or c <= max_count)
    ]
    qn = _norm(q, False) if q else ""
    minor = [(k, c) for k, c in minor_all if qn in k] if qn else minor_all
    m_ans = sum(c for _, c in minor_all)

    def pct(a, b):
        return round(a / b * 100, 1) if b else 0.0

    return dict(
        title=label,
        kind=kind,
        n=len(d),
        base=base,
        kinds=len(counts),
        answers=total,
        top=top,
        max_count=max_count,
        pref_only=bool(pref_only and kind == "text"),
        majority=dict(
            labels=[shown[k] for k, _ in major],
            counts=[c for _, c in major],
            shares=[pct(c, base) for _, c in major],
            coverage=pct(sum(c for _, c in major), total),
        ),
        minority=dict(
            kinds=len(minor_all),
            kinds_share=pct(len(minor_all), len(counts)),
            answers=m_ans,
            answers_share=pct(m_ans, total),
            matched=len(minor),
            truncated=len(minor) > limit,
            items=[dict(text=shown[k], count=c) for k, c in minor[:limit]],
        ),
    )
