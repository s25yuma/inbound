# app.py  --  インバウンド消費動向（B1地域調査・個票）可視化アプリ
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_CSV = "個票データ2024Q1_2025Q4_sel_col.csv"
MIN_N = 5          # 平均値を表示する最小サンプル数（未満は「データ不足」）
FIRST_CODE = 57    # 調査回57 = 2024年4-6月期

app = Flask(__name__, template_folder="templates")
app.json.ensure_ascii = False

# =====================================================
# マスタ（変数リストの選択肢ラベル）

# =====================================================\
LABELS = {
    "arrange":   {1: "団体ツアー", 2: "個人パッケージ", 3: "個別手配"},
    "jr":        {1: "JR PASS利用", 2: "利用なし"},
    "status":    {1: "短期滞在(90日以内)", 2: "医療滞在", 3: "長期滞在(91日以上)", 4: "デジタルノマド",
                  5: "トランジット(外出あり)", 6: "トランジット(外出なし)", 7: "永住者・配偶者等", -98: "不明"},
    "companion": {1: "自分ひとり", 2: "夫婦・パートナー", 3: "家族・親族", 4: "職場の同僚", 5: "友人", 6: "その他"},
    "visits":    {1: "1回目", 2: "2回目", 3: "3回目", 4: "4回目", 5: "5回目", 6: "6～9回目", 7: "10～19回", 8: "20回以上"},
    "purpose":   {1: "観光・レジャー", 2: "親族・知人訪問", 3: "ハネムーン", 4: "学校関連", 5: "スポーツ", 6: "イベント",
                  7: "留学", 8: "治療・検診", 9: "インセンティブ", 10: "展示会", 11: "国際会議", 12: "企業ミーティング",
                  13: "研修", 14: "その他ビジネス", 15: "トランジット", 16: "その他"},
    "stay_c":    {1: "3日以内", 2: "4～6日", 3: "7～13日", 4: "14～20日", 5: "21～27日", 6: "28～90日", 7: "91日以上"},
    "product":   {1: "菓子類", 2: "酒類", 3: "生鮮農産物", 4: "その他食料品・飲料", 5: "化粧品・香水", 6: "医薬品",
                  7: "健康グッズ・トイレタリー", 8: "衣類", 9: "靴・かばん・革製品", 10: "電気製品", 11: "時計・カメラ",
                  12: "宝石・貴金属", 13: "民芸品・伝統工芸品", 14: "本・雑誌", 15: "ソフトウェア", 16: "その他買物代", 17: "その他"},
}
AGE = {"15_19": "15～19歳", "20_29": "20～29歳", "30_39": "30～39歳", "40_49": "40～49歳",
       "50_59": "50～59歳", "60_69": "60～69歳", "70ov": "70歳以上"}

MULTI = {  # 複数選択フラグ: (列接頭辞, ラベル)
    "transport": ("D1", ["航空", "新幹線", "鉄道・モノレール", "貸切バス", "長距離バス", "近郊バス", "タクシー・ハイヤー",
                         "レンタカー", "レンタサイクル", "船舶", "その他", "徒歩のみ"]),
    "shop":      ("D2", ["百貨店", "家電量販店", "ファッション専門店", "100円ショップ", "SA・道の駅", "観光地の土産店",
                         "宿泊施設", "スーパー", "コンビニ", "ドラッグストア", "ディスカウントストア", "アウトレット",
                         "都心の複合商業施設", "その他SC", "駅構内の店舗", "空港の免税店", "クルーズ港湾内の店舗",
                         "その他", "買い物しなかった"]),
    "f1_reason": ("F1", ["美味しい", "品質が良い", "自国で入手が難しい", "価格が手頃", "日本製", "伝統的・日本独特",
                         "好きなブランド", "デザインが良い", "人気・有名", "お土産に良い", "便利", "その他"]),
    "f2_reason": ("F2", ["美味しい", "食材が新鮮", "自国で味わえない", "価格が手頃", "量・種類が適切", "伝統的・日本独特",
                         "好きな料理", "見た目が良い", "人気・有名", "健康に良い", "店のサービス・雰囲気", "その他"]),
}

# 支出内訳（山形県内・円/人）。総額 = 宿泊+飲食+域内交通+娯楽+買物+その他（域間交通は総額に含まれない）
SPEND_PARTS = [("宿泊費", "C4a_pref_6"), ("飲食費", "C4b_pref_6"), ("域内交通費", "C4域内交通_pref_6"),
               ("娯楽等サービス", "C4i_pref_6"), ("買物代", "C4j_pref_6"), ("その他", "C4k_pref_6")]
SPEND_REF = ("域間交通費※総額に含まず", "C4域間交通_pref_6")


def quarter_label(code):
    q = 2024 * 4 + 1 + (int(code) - FIRST_CODE)
    y, i = divmod(q, 4)
    return f"{y}年{i * 3 + 1}-{i * 3 + 3}月期"


def read_csv(name):
    path = os.path.join(DATA_DIR, name)
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"文字コードを判別できません: {name}")


# =====================================================
# 前処理
# =====================================================
def load_data():
    raw = read_csv(MAIN_CSV)
    country_m = read_csv("国籍地域.csv").set_index("code")["country"]
    sexage_m = read_csv("性年代.csv").set_index("code")["sexAge"]
    ccol = [c for c in raw.columns if c.startswith("国籍")][0]

    d = pd.DataFrame(index=raw.index)
    d["quarter"] = raw["調査回"].map(quarter_label)
    d["quarter_code"] = raw["調査回"]
    d["country"] = raw[ccol].map(country_m)
    sa = raw["性年代"].map(sexage_m)
    d["sex"] = sa.str[0].map({"m": "男性", "f": "女性"})
    d["age"] = sa.str[1:].map(AGE)
    d["sexage"] = d["sex"] + d["age"]
    for key, col in [("arrange", "B1"), ("jr", "B3"), ("status", "E3"), ("companion", "E5"),
                     ("visits", "E6"), ("purpose", "E7"), ("stay_c", "滞在日数_c"), ("product", "F1_c")]:
        d[key] = raw[col].map(LABELS[key])           # 未定義コード(-98等)は欠測扱い

    # 数値: -99(非該当)/-98(不明) の扱い
    parts = [c for _, c in SPEND_PARTS] + [SPEND_REF[1], "C4総額_pref_6"]
    for c in parts:  # 支出回答者(欠測でない行)では -99 は「支出なし=0」
        d[c] = raw[c].where(raw[c].isna(), raw[c].clip(lower=0))
    d["spend_total"] = d["C4総額_pref_6"]
    d["nights_pref"] = raw["C3_pref_6"].where(raw["C3_pref_6"] >= 0)
    d["stay_days"] = raw["滞在日数"].where(raw["滞在日数"] > 0)
    d["b2"] = raw["B2"].where(raw["B2"] > 0)
    d["repeat"] = (raw["E6"] >= 2).astype(float)

    for _, (pre, labels) in MULTI.items():
        for i in range(1, len(labels) + 1):
            d[f"{pre}_{i}"] = raw[f"{pre}_{i}"].where(raw[f"{pre}_{i}"] >= 0)

    # 山形県内の訪問スポット / 入国空港 / 自由記述
    c1 = [f"C1_{i}" for i in range(1, 14)]
    d["spots"] = raw[c1].apply(lambda r: sorted({re.sub(r"\s*\[山形県\]$", "", v).strip() for v in r
                                                 if isinstance(v, str) and v.endswith("[山形県]") and not v.startswith("その他")}), axis=1)
    d["entry"] = raw["C1_入国"].str.replace(r"\s*\[.*\]", "", regex=True)
    for k in ("F1", "F2"):
        s = raw[k].astype(str).str.strip()
        d[k.lower() + "_text"] = s.where(~s.isin(["", "nan", "回答なし"]))
    return d


DATA = load_data()

# 軸にできるカテゴリ変数: key -> (表示名, 並び順(None=件数順))
CAT_VARS = {
    "quarter": ("調査期", [quarter_label(c) for c in range(FIRST_CODE, FIRST_CODE + 8)]),
    "country": ("国籍・地域", None),
    "sex": ("性別", ["男性", "女性"]),
    "age": ("年代", list(AGE.values())),
    "sexage": ("性年代", [s + a for s in ("男性", "女性") for a in AGE.values()]),
    "arrange": ("旅行手配方法", list(LABELS["arrange"].values())),
    "jr": ("JRパス", list(LABELS["jr"].values())),
    "status": ("在留資格", list(LABELS["status"].values())),
    "companion": ("同行者", list(LABELS["companion"].values())),
    "visits": ("来訪回数", list(LABELS["visits"].values())),
    "purpose": ("来訪目的", list(LABELS["purpose"].values())),
    "stay_c": ("滞在日数区分", list(LABELS["stay_c"].values())),
    "product": ("最も満足した購入商品", None),
}
METRICS = {
    "count": ("回答数", "人", None), "share": ("構成比", "%", None),
    "spend": ("平均支出（山形県内）", "円", "spend_total"),
    "nights": ("平均泊数（山形県内）", "泊", "nights_pref"),
    "stay": ("平均滞在日数（日本全体）", "日", "stay_days"),
    "b2": ("平均パッケージ・航空券料金", "円", "b2"),
}


# =====================================================
# 補助関数
# =====================================================
def num(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def filtered():
    d = DATA
    for k in ("quarter", "country", "sex", "age"):
        v = request.args.get(k)
        if v:
            d = d[d[k] == v]
    return d


def order_of(d, key, limit=None):
    present = d[key].dropna()
    fixed = CAT_VARS[key][1]
    labels = [l for l in fixed if l in set(present)] if fixed else present.value_counts().index.tolist()
    return labels[:limit] if limit else labels


def chart(title, labels, series, unit="", kind="bar", horizontal=False, stacked=False, note="", wide=False):
    """series: [(名前, 値リスト, nリスト|None)]"""
    return dict(title=title, type=kind, labels=labels, unit=unit, horizontal=horizontal, stacked=stacked,
                note=note, wide=wide,
                datasets=[dict(label=n, data=[num(v) for v in vals], n=ns) for n, vals, ns in series])


def mean_by(d, key, col, labels):
    g = d.groupby(key)[col].agg(["mean", "count"]).reindex(labels)
    vals = g["mean"].where(g["count"] >= MIN_N)
    return vals.tolist(), g["count"].fillna(0).astype(int).tolist()


def count_chart(d, key, title, top=None, **kw):
    s = d[key].dropna().value_counts()
    labels = order_of(d, key, top)
    n = s.reindex(labels).fillna(0).astype(int)
    return chart(title, labels, [("回答数", n.tolist(), None)], unit="人", **kw)


def multi_chart(d, name, title, top=12):
    pre, labels = MULTI[name]
    rows = []
    for i, lab in enumerate(labels, 1):
        s = d[f"{pre}_{i}"].dropna()
        if len(s):
            rows.append((lab, s.mean() * 100, len(s)))
    rows = sorted(rows, key=lambda r: -r[1])[:top]
    return chart(title, [r[0] for r in rows], [("割合", [r[1] for r in rows], [r[2] for r in rows])],
                 unit="%", horizontal=True, note="複数回答。回答者に対する割合")


def rank_chart(values, title, total, top=15, note=""):
    cnt = Counter(values).most_common(top)
    return chart(title, [k for k, _ in cnt], [("割合", [v / total * 100 for _, v in cnt], [v for _, v in cnt])],
                 unit="%", horizontal=True, note=note)


# =====================================================
# ルーティング
# =====================================================
@app.route("/")
def index():
    meta = dict(
        quarters=CAT_VARS["quarter"][1],
        countries=order_of(DATA, "country"),
        sexes=CAT_VARS["sex"][1], ages=CAT_VARS["age"][1],
        vars={k: v[0] for k, v in CAT_VARS.items()},
        metrics={k: v[0] for k, v in METRICS.items()},
    )
    return render_template("dashboard.html", meta=meta)


@app.route("/api/dashboard")
def api_dashboard():
    d = filtered()
    n = len(d)
    if n == 0:
        return jsonify(n=0, kpis=[], sections=[])
    sp = d["spend_total"].dropna()
    kpis = [
        ("回答数", f"{n:,}", "人"),
        ("平均支出（山形県内）", f"{sp.mean():,.0f}" if len(sp) else "-", "円/人"),
        ("支出あり回答者の平均", f"{sp[sp > 0].mean():,.0f}" if (sp > 0).any() else "-", "円/人"),
        ("平均泊数（山形県内）", f"{d['nights_pref'].mean():.1f}" if d["nights_pref"].notna().any() else "-", "泊"),
        ("平均滞在日数（日本）", f"{d['stay_days'].mean():.1f}" if d["stay_days"].notna().any() else "-", "日"),
        ("リピーター比率", f"{d['repeat'].mean() * 100:.0f}", "%"),
    ]

    # --- 属性
    attr = [count_chart(d, "country", "国籍・地域", top=12, horizontal=True),
            count_chart(d, "sexage", "性年代"),
            count_chart(d, "companion", "同行者"), count_chart(d, "visits", "来訪回数"),
            count_chart(d, "purpose", "来訪目的", top=8, horizontal=True),
            count_chart(d, "arrange", "旅行手配方法")]

    # --- 支出
    sample = d[d["spend_total"].notna()]
    names = [p for p, _ in SPEND_PARTS] + [SPEND_REF[0]]
    cols = [c for _, c in SPEND_PARTS] + [SPEND_REF[1]]
    allm = [sample[c].mean() for c in cols]
    payer = [sample.loc[sample[c] > 0, c].mean() for c in cols]
    spend = [chart("費目別 平均支出（山形県内）", names,
                   [("全回答者平均（0円含む）", allm, None), ("支出者のみの平均", payer, None)],
                   unit="円", horizontal=True,
                   note=f"支出回答者 n={len(sample)}。総額＝域間交通費を除く6費目の合計")]
    q = order_of(d, "quarter")
    vals, ns = mean_by(d, "quarter", "spend_total", q)
    spend.append(chart("調査期別 平均支出（山形県内）", q, [("平均支出", vals, ns)], unit="円", kind="line",
                       note=f"n<{MIN_N} は非表示"))
    top_c = [c for c in order_of(d, "country") if (sample["country"] == c).sum() >= 15][:8]
    if top_c:
        series = [(p, [sample.loc[sample["country"] == c, col].mean() for c in top_c], None)
                  for p, col in SPEND_PARTS]
        spend.append(chart("国籍・地域別 平均支出の内訳", top_c, series, unit="円", stacked=True,
                           note="支出回答者15人以上の国・地域のみ", wide=True))
    nt = d["nights_pref"].dropna().clip(upper=7).astype(int)
    nl = ["日帰り"] + [f"{i}泊" for i in range(1, 7)] + ["7泊以上"]
    spend.append(chart("山形県内の泊数", nl, [("回答数", [int((nt == i).sum()) for i in range(8)], None)], unit="人"))

    # --- 行動
    behavior = [multi_chart(d, "transport", "日本滞在中の交通手段"), multi_chart(d, "shop", "買物場所"),
                rank_chart([s for lst in d["spots"] for s in lst], "山形県内の訪問スポット（上位15）", n,
                           note="C1訪問地のうち山形県内。「その他山形県」は除く"),
                rank_chart(d["entry"].dropna(), "入国空港・港（上位10）", n, top=10)]

    # --- 満足
    sat = [count_chart(d, "product", "最も満足した購入商品", top=10, horizontal=True),
           multi_chart(d, "f1_reason", "購入商品に満足した理由"),
           multi_chart(d, "f2_reason", "飲食に満足した理由"),
           rank_chart(d["f2_text"].dropna(), "最も満足した飲食（自由記述・上位）", n, note="表記ゆれは未統合"),
           rank_chart(d["f1_text"].dropna(), "最も満足した購入商品（自由記述・上位）", n, note="表記ゆれは未統合")]

    sections = [dict(title="回答者属性", charts=attr), dict(title="消費額（山形県内）", charts=spend),
                dict(title="行動・訪問地", charts=behavior), dict(title="満足度・購入／飲食", charts=sat)]
    return jsonify(n=n, kpis=[dict(label=a, value=b, unit=c) for a, b, c in kpis], sections=sections)


@app.route("/api/explore")
def api_explore():
    d = filtered()
    x, g, m = request.args.get("x", "country"), request.args.get("group") or None, request.args.get("metric", "count")
    if x not in CAT_VARS or m not in METRICS or (g and (g not in CAT_VARS or g == x)):
        abort(400)
    labels = order_of(d, x, 15)
    groups = order_of(d, g, 8) if g else [None]
    series = []
    for gl in groups:
        sub = d if gl is None else d[d[g] == gl]
        sub = sub[sub[x].notna()]
        if m in ("count", "share"):
            s = sub.groupby(x).size().reindex(labels, fill_value=0)
            ns = s.tolist()
            vals = (s / s.sum() * 100 if m == "share" and s.sum() else s).tolist()
        else:
            vals, ns = mean_by(sub, x, METRICS[m][2], labels)
        series.append((gl or METRICS[m][0], vals, ns))
    title = f"{CAT_VARS[x][0]}別 {METRICS[m][0]}" + (f"（{CAT_VARS[g][0]}別）" if g else "")
    note = f"平均値はn<{MIN_N}を非表示。構成比は系列ごとに合計100%" if m != "count" else ""
    return jsonify(chart(title, labels, series, unit=METRICS[m][1], note=note))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
