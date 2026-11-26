# textmining.py -- 自由記述（F1買い物／F2飲食／F3良かった体験）のテキストマイニング
# test6.ipynb の処理（山形判定・県外マーカー・保護語・表記ゆれ・TF-IDF）をWebアプリ用に移植したもの。
# 形態素解析(Janome)は初回アクセス時に1回だけ実行して結果をキャッシュする。
import math
import re
import unicodedata
from collections import Counter

import pandas as pd

try:  # Janome が無い環境でも動くようにする（その場合は区切り単位の簡易集計）
    from janome.tokenizer import Tokenizer

    _TOKENIZER = Tokenizer()
except Exception:  # pragma: no cover
    _TOKENIZER = None
HAS_JANOME = _TOKENIZER is not None

TARGETS = {  # DATA の列名 -> 表示名
    "f1_text": "買い物（F1）",
    "f2_text": "飲食（F2）",
    "f3_text": "良かった体験（F3）",
}

# =========================================================
# 設定（ノートブックと同じ。足りない語はここに追加する）
# =========================================================
NO_ANSWER = {"回答なし", "非該当", "無回答", "特になし", "なし", "不明", ""}

# 山形県内と判断するための語（「蔵王」単独は宮城県側の可能性があるため含めない）
YAMAGATA_MARKERS = [
    "山形県", "山形", "米沢", "山寺", "立石寺", "銀山温泉", "銀山", "尾花沢", "天童", "寒河江", "上山",
    "鶴岡", "酒田", "庄内", "新庄", "最上", "置賜", "村山", "月山", "羽黒山", "出羽三山", "鳥海山",
    "さくらんぼ", "サクランボ", "芋煮", "だだちゃ豆", "玉こんにゃく", "米沢牛", "山形牛", "出羽桜",
    "蔵王温泉", "山形蔵王", "山形藏王",
]
# 同じ区切りにこれがあれば山形県内として採用しない
OUTSIDE_MARKERS = [
    "宮城", "仙台", "松島", "白石", "鳴子", "作並", "秋保", "石巻", "塩竈", "気仙沼", "キツネ村", "利久", "善治郎",
    "ずんだ", "笹かま", "御釜", "蔵王エコーライン", "十和田湖", "奥入瀬", "八甲田山", "青森", "秋田",
    "北海道", "京都", "長野", "白川郷", "大内宿", "只見線", "猊鼻渓", "牛タン専門店",
]
# 形態素解析で分割させたくない語
PROTECTED_WORDS = [
    "蔵王温泉", "銀山温泉", "山形蔵王", "山形藏王", "出羽三山", "羽黒山", "鳥海山", "米沢牛", "山形牛",
    "玉こんにゃく", "だだちゃ豆", "さくらんぼ", "サクランボ", "立石寺", "山寺", "出羽桜", "天童温泉", "寒河江", "尾花沢",
]
# 表記ゆれの統一
SYNONYMS = {"銀山": "銀山温泉", "立石寺": "山寺", "山形藏王": "山形蔵王", "サクランボ": "さくらんぼ"}
# 集計に出さない一般語
STOPWORDS = {
    "山形", "山形県", "日本", "旅行", "観光", "こと", "もの", "とても", "全て", "すべて", "今回", "次回",
    "食事", "夕食", "朝食", "昼食", "ランチ", "ディナー", "回答", "場所", "感じ", "体験", "訪問",
}


def normalize_text(text):
    """全角・半角などの表記をそろえる（全角括弧は半角になる）"""
    return unicodedata.normalize("NFKC", str(text)).strip()


YAMAGATA_NORM = [normalize_text(w) for w in YAMAGATA_MARKERS]
OUTSIDE_NORM = [normalize_text(w) for w in OUTSIDE_MARKERS]
PROTECTED_NORM = sorted({normalize_text(w) for w in PROTECTED_WORDS}, key=len, reverse=True)
SYNONYMS_NORM = {normalize_text(k): normalize_text(v) for k, v in SYNONYMS.items()}
STOPWORDS_NORM = {normalize_text(w) for w in STOPWORDS}
NO_ANSWER_NORM = {normalize_text(w) for w in NO_ANSWER}

SEPARATORS = set("、,。．.\n\r/／・;；")
OPEN_BRACKETS, CLOSE_BRACKETS = set("([【"), set(")]】")
PAREN_RE = re.compile(r"[(\[【]([^()\[\]【】]*)[)\]】]")
STRAY_BRACKET_RE = re.compile(r"[()\[\]【】]")


def is_answer(value):
    return isinstance(value, str) and normalize_text(value) not in NO_ANSWER_NORM


def split_segments(text):
    """区切り文字で分割する。括弧の中の区切り文字では分割しない（饅頭(銀山温泉、米沢) を1つにする）"""
    text = normalize_text(text)
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch in OPEN_BRACKETS:
            depth += 1
        elif ch in CLOSE_BRACKETS:
            depth = max(0, depth - 1)
        if depth == 0 and ch in SEPARATORS:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for part in parts for p in re.split(r"\s{2,}", part) if p.strip()]


def is_yamagata_segment(segment):
    if any(w in segment for w in OUTSIDE_NORM):
        return False
    return any(w in segment for w in YAMAGATA_NORM)


def remove_place_annotations(segment):
    """括弧内に山形の地名・名物があれば括弧ごと除く。例: 饅頭(銀山温泉) -> 饅頭"""

    def repl(m):
        return " " if any(w in m.group(1) for w in YAMAGATA_NORM) else " " + m.group(1) + " "

    return STRAY_BRACKET_RE.sub(" ", PAREN_RE.sub(repl, segment))


def extract_words(segment):
    """区切りから名詞を抽出（保護語・表記ゆれ・ストップワードを処理）"""
    segment = remove_place_annotations(segment)
    words, remaining = [], segment
    for pw in PROTECTED_NORM:
        if pw in remaining:
            words.append(pw)
            remaining = remaining.replace(pw, " ")
    if _TOKENIZER is not None:
        for token in _TOKENIZER.tokenize(remaining):
            pos = token.part_of_speech.split(",")
            if pos[0] != "名詞" or pos[1] in ("非自立", "代名詞", "数", "接尾"):
                continue
            if len(token.surface) >= 2:
                words.append(token.surface)
    else:  # 簡易モード: 空白区切りの語句をそのまま使う
        words.extend(w for w in remaining.split() if len(w) >= 2)
    out = []
    for w in words:
        w = SYNONYMS_NORM.get(w, w)
        if w not in STOPWORDS_NORM:
            out.append(w)
    return out


def answer_words(text, yamagata_only):
    """1人の回答から単語の集合を作る（同じ語は1人1回）。戻り値: (単語集合, 採用した区切り数)"""
    words, used = set(), 0
    for seg in split_segments(text):
        if yamagata_only and not is_yamagata_segment(seg):
            continue
        used += 1
        words.update(extract_words(seg))
    return frozenset(words), used


def compute_tfidf(counters):
    """F1・F2・F3を1文書ずつとみなしたTF-IDF（平滑化はsklearnと同じ）"""
    n_docs = len(counters)
    df_count = Counter()
    for c in counters.values():
        df_count.update(c.keys())
    out = {}
    for col, c in counters.items():
        total = sum(c.values())
        out[col] = {
            w: (cnt / total) * (math.log((1 + n_docs) / (1 + df_count[w])) + 1) for w, cnt in c.items()
        } if total else {}
    return out


def canon(word):
    w = normalize_text(word)
    return SYNONYMS_NORM.get(w, w)


class TextMining:
    """DATA（DataFrame）に対する集計。形態素解析の結果は (設問, 山形のみ) ごとにキャッシュする。"""

    def __init__(self, data):
        self.data = data
        self._cache = {}

    def _table(self, topic, yamagata_only):
        key = (topic, bool(yamagata_only))
        if key not in self._cache:
            sets, used = [], []
            for v in self.data[topic]:
                w, u = answer_words(v, yamagata_only) if is_answer(v) else (frozenset(), 0)
                sets.append(w)
                used.append(u)
            self._cache[key] = (pd.Series(sets, index=self.data.index), pd.Series(used, index=self.data.index))
        return self._cache[key]

    def _answered(self, d, topic):
        return d[topic].map(is_answer)

    def counter(self, d, topic, yamagata_only):
        sets, used = self._table(topic, yamagata_only)
        c = Counter()
        for s in sets.loc[d.index]:
            c.update(s)
        return c, int(self._answered(d, topic).sum()), int(used.loc[d.index].sum())

    def summary(self, d, topic, yamagata_only=True, mode="count", top=40):
        topics = list(TARGETS) if mode == "tfidf" else [topic]
        got = {t: self.counter(d, t, yamagata_only) for t in topics}
        c, n_ans, n_used = got[topic]
        scores = compute_tfidf({t: v[0] for t, v in got.items()})[topic] if mode == "tfidf" else dict(c)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        words = [
            dict(word=w, score=round(s, 4), count=c[w], rate=round(c[w] / n_ans * 100, 1) if n_ans else 0.0)
            for w, s in ranked
        ]
        return dict(words=words, answers=n_ans, segments=n_used, kinds=len(c), mode=mode, has_janome=HAS_JANOME)

    def kwic(self, d, topic, word, yamagata_only=True, limit=80):
        """その語を含む回答の原文を返す（語の確認用）"""
        word = canon(word)
        sets, _ = self._table(topic, yamagata_only)
        sets = sets.loc[d.index]
        hits = [word] + [k for k, v in SYNONYMS_NORM.items() if v == word]
        rows = []
        for idx in d.index[sets.map(lambda s: word in s).values]:
            r = d.loc[idx]
            rows.append(dict(text=normalize_text(r[topic]), quarter=r.get("quarter"), country=r.get("country")))
        return dict(word=word, total=len(rows), hits=hits, items=rows[:limit])

    def trend(self, d, axis, labels, topic, words, yamagata_only=True, metric="rate", min_n=10):
        """軸（調査期・年代など）の区分ごとに、各語の出現率(言及者÷回答者)または人数を返す"""
        sets, _ = self._table(topic, yamagata_only)
        sets = sets.loc[d.index]
        answered = self._answered(d, topic).values
        series = [dict(label=w, values=[], mentions=[], n=[]) for w in words]
        for lab in labels:
            m = (d[axis] == lab).values & answered
            sub = sets[m]
            n = int(m.sum())
            for s, w in zip(series, words):
                cnt = int(sub.map(lambda x: w in x).sum()) if n else 0
                s["mentions"].append(cnt)
                s["n"].append(n)
                if metric == "count":
                    s["values"].append(cnt)
                else:
                    s["values"].append(round(cnt / n * 100, 1) if n >= min_n else None)
        return series
