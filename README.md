# RAF Playground — 3手法を1つのCLIで切り替える検索拡張時系列予測

Chronos-Bolt をバックボーンに、**素のChronos / コンテキスト拡張RAF / Cross-RAG** の3手法を
`--method` で切り替えて学習・評価できる軽量フレームワーク。`cross-rag/` のモデル思想を引き継ぎつつ、
データローダ・検索は依存の軽い自前実装にしている。

| `--method` | 手法 | 検索 | 融合のしかた | 学習対象 | 出典 |
|---|---|---|---|---|---|
| `none` | 素の Chronos（zero-shot） | なし | なし | なし | Chronos-Bolt |
| `raf` | コンテキスト拡張 RAF | top-k（既定は連結） | 検索窓を整合してコンテキストの前に**連結** | naive=凍結 / advanced=backbone微調整 | Tire et al. 2026 |
| `cross_raf` | Cross-RAG | top-k | **cross-attention + self-attention** で融合 | 融合モジュールのみ（backbone凍結） | Lee et al. 2026 |

バックボーンは3手法とも Chronos-Bolt で共通。違いは「検索の有無/件数・融合方式・学習対象」だけで、
`methods/` の各ファイルに閉じ込めてある。

---

## セットアップ

```bash
pip3 install -r requirements.txt        # 失敗時は pip3 install --break-system-packages -r requirements.txt
```

- 依存：`torch` / `transformers` / `chronos-forecasting` / `faiss-cpu` / `numpy` / `pandas`
- バックボーンは初回実行時に Hugging Face から自動取得（既定 `amazon/chronos-bolt-small`）。
- faiss と torch の OpenMP 競合回避は `_bootstrap.py` が自動で行う（後述）。

---

## データ

`--root-path` 直下に `date` 列＋変数列を持つ標準ベンチCSVを置く。各変数列を独立した1変量系列として扱い、
Informer流の train/val/test 分割・チャネル標準化（train統計でz-score）を行う。
**test = クエリ、train = 検索DB**（`--retrieval-split` で変更可）。

同梱データ（`./Datasets/`）:

| 指定例 | パス | `--dataset` 名 | 分割 |
|---|---|---|---|
| ETT 時間足 | `./Datasets/ETT-small/ETTh1.csv` (h1/h2) | `ETTh1` `ETTh2` | ETT専用境界（12/4/4ヶ月）|
| ETT 分足 | `./Datasets/ETT-small/ETTm1.csv` (m1/m2) | `ETTm1` `ETTm2` | ETT専用境界 |
| weather | `./Datasets/weather/weather.csv` | `custom`（任意名可） | 0.7/0.1/0.2 |
| exchange_rate | `./Datasets/exchange_rate/exchange_rate.csv` | `custom` | 0.7/0.1/0.2 |
| electricity | `./Datasets/electricity/electricity.csv` | `custom` | 0.7/0.1/0.2 |

`--dataset` は分割境界の決定にのみ使う（`ETTh1/h2/m1/m2` だけ専用境界、それ以外は 0.7/0.1/0.2）。

---

## クイックスタート

### 1) 素の Chronos（学習不要）
```bash
python3 evaluate.py --method none \
  --dataset ETTh1 --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv \
  --seq-len 512 --pred-len 64
```

### 2) RAF naive（学習不要・top-1連結）
```bash
python3 evaluate.py --method raf --raf-mode naive \
  --dataset ETTh1 --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv \
  --top-k 1
```

### 2') RAF advanced（backbone微調整 → 評価）
```bash
python3 train.py    --method raf --raf-mode advanced --dataset ETTh1 \
  --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv --top-k 1 --lr 1e-4
python3 evaluate.py --method raf --raf-mode advanced --dataset ETTh1 \
  --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv --top-k 1 \
  --checkpoint ./checkpoints/raf_ETTh1/best.pth
```

### 3) Cross-RAG（融合モジュール学習 → 評価）
```bash
python3 train.py    --method cross_raf --dataset ETTh1 \
  --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv --top-k 15
python3 evaluate.py --method cross_raf --dataset ETTh1 \
  --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv --top-k 15 \
  --checkpoint ./checkpoints/cross_raf_ETTh1/best.pth
```

> `none` と `raf --raf-mode naive` は学習不要。`train.py` を実行しても「学習不要」と表示して終了する。

---

## オプション一覧

### 共通（train.py / evaluate.py 両方）

| オプション | 既定 | 説明 |
|---|---|---|
| `--method` | `none` | `none` / `raf` / `cross_raf` |
| `--raf-mode` | `naive` | `--method raf` 時のみ。`naive`=凍結 / `advanced`=backbone微調整 |
| `--root-path` | `./Datasets/ETT-small/` | CSV があるディレクトリ |
| `--data-path` | `ETTh1.csv` | CSV ファイル名 |
| `--dataset` | `ETTh1` | 分割境界の決定名（`ETTh1/h2/m1/m2` のみ専用境界）|
| `--features` | `M` | `M`=全変数列を独立系列 / `S`=`--target` の1列のみ |
| `--target` | `OT` | `--features S` のときの対象列名 |
| `--seq-len` | `512` | コンテキスト長（≤ backbone context_length=2048）|
| `--pred-len` | `64` | 予測長（≤ backbone prediction_length=64、**cross_raf は ==64 必須**）|
| `--no-scale` | off | チャネル標準化を無効化（生スケールで評価）|
| `--chronos-model` | `amazon/chronos-bolt-small` | backbone。`-tiny/-mini/-small/-base` または local path |
| `--top-k` | `15` | 検索件数。cross_raf=融合本数 / raf=連結本数（context長に自動cap、rafは1〜3推奨）|
| `--retrieval-metric` | `cosine` | faiss類似度：`cosine` / `euclidean` / `correlation`(Pearson) |
| `--retrieval-split` | `train` | 検索DBを作る分割 |
| `--retrieval-stride` | `1` | 検索DBの窓スライド幅（1=論文どおり全窓 / 大きくすると件数・メモリ減）|
| `--augment-mode` | `moe` | cross_raf 融合モード（`moe`=cross+self）|
| `--mix-lambda` | `0.7` | cross_raf の cross/self 混合比 |
| `--output-dir` | `./checkpoints` | チェックポイント / 結果の保存先 |
| `--gpu` | `0` | GPU id（cuda時）。mps/cpuは自動判定 |
| `--batch-size` | `256` | バッチサイズ |
| `--num-workers` | `2` | DataLoader ワーカ数 |
| `--seed` | `2021` | 乱数シード |

### train.py 専用

| オプション | 既定 | 説明 |
|---|---|---|
| `--train-steps` | `10000` | optimizerステップ数（Cross-RAG論文 Table A.2）。step駆動でローダをcycleして必ず到達 |
| `--lr` | `3e-4` | **定数**学習率（Cross-RAG論文値・スケジューラなし。raf advanced は 1e-5〜1e-4 推奨）|
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--grad-clip` | `1.0` | 勾配クリップ |
| `--train-stride` | `1` | 学習窓のスライド幅 |
| `--save-freq` | `2000` | 何ステップごとに保存するか |
| `--resume` | `""` | 再開する重みのパス |

### evaluate.py 専用

| オプション | 既定 | 説明 |
|---|---|---|
| `--checkpoint` | `""` | 学習済み重み（cross_raf / raf advanced で指定）|
| `--eval-stride` | `pred_len` | test窓のスライド幅。`1` で全窓（完全ベンチ再現・低速）|
| `--result-file` | `result.txt` | `--output-dir` 下に追記する結果ファイル名 |

### 固定パラメータ（CLIに出していないコード内デフォルト）

CLIオプション以外に、コード内で固定している主な既定値：

**backbone（Chronos-Bolt のモデル設定。全サイズ共通、`d_model`のみ異なる）**

| 値 | デフォルト | 備考 |
|---|---|---|
| `context_length` | `2048` | `--seq-len` の上限 |
| `prediction_length` | `64` | `--pred-len` の上限（cross_raf は ==64）|
| 分位点 | `[0.1, 0.2, …, 0.9]`（9点）| 点予測は中央値(0.5)を使用 |
| patch size / stride | `16 / 16` | |
| `d_model` | tiny=256 / mini=384 / **small=512** / base=768 | `--chronos-model` で決まる |

**cross_raf 融合ヘッド（`models/chronos/retrieval_model.py`、固定）**

| 値 | デフォルト |
|---|---|
| attention ヘッド数（cross/self とも） | `8` |
| dropout | `0.2` |
| encoder MLP / FFN | `Linear→ReLU→Linear`（hidden=`d_model`）|
| `INPUT_LEN`（encode_mlp_x の入力長） | `--seq-len` から設定 |
| `LAMBDA`（cross/self 混合比） | `--mix-lambda`（既定0.7）から設定 |
| 学習対象パラメータ数 | 約4M（small、backbone凍結時）|

**学習・最適化（`train.py`、固定挙動）**

| 値 | デフォルト |
|---|---|
| optimizer | `AdamW`（`--lr` `--weight-decay`）|
| 学習率スケジュール | **なし（定数LR）** — Cross-RAG論文 Table A.2 準拠 |
| 学習制御 | step駆動（ローダをcycleし `--train-steps` まで）|
| 損失 | Chronos純正の quantile regression loss |
| 勾配クリップ | `--grad-clip`（既定1.0）|

**検索・前処理（固定）**

| 項目 | デフォルト |
|---|---|
| faiss index | `IndexFlatIP`(cosine/correlation) / `IndexFlatL2`(euclidean)（厳密検索）|
| 検索キー正規化 | min-max → metric正規化（cosine=L2 / correlation=平均除去+L2 / euclidean=なし）|
| self-match 除外 | 距離 `<1e-6` を除外（学習クエリのみ。評価は全件保持）|
| チャネル標準化 | train統計で z-score（eps `1e-8`、`--no-scale`で無効）|
| 分割境界 | ETTh1/h2/m1/m2=12/4/4ヶ月、それ以外=0.7/0.1/0.2 |
| 評価空間 | 標準化空間で MSE/MAE |

---

## ファイルの役割

```
.
├── train.py            学習エントリ。method取得→データ/検索器構築→学習ループ→保存
├── evaluate.py         評価エントリ。test窓を予測→MSE/MAE算出→結果追記
├── config.py           argparse一括定義（共通 + train専用 + eval専用）
├── _bootstrap.py       faiss/torch の OpenMP二重ロード回避。各エントリで最初にimport
├── requirements.txt
│
├── methods/            ★手法ごとのロジック（手法追加はここに1ファイル＋registry登録）
│   ├── registry.py     "none"/"raf"/"cross_raf" → Methodクラス
│   ├── base.py         共通インターフェース（build_model/compute_loss/predict/needs_*）
│   ├── _backbone.py    Chronos-Boltロード補助 + 分位点→点予測(median)抽出
│   ├── chronos_base.py none      : 凍結Chronos。検索なしでそのまま予測
│   ├── raf.py          raf       : top-k検索→整合連結(build_raf_context)→予測/学習
│   └── cross_raf.py    cross_raf : 融合ヘッド構築・freeze制御→top-k融合で予測/学習
│
├── models/
│   └── chronos/
│       ├── __init__.py          公式chronosのplainモデルを再export + 融合サブクラス公開
│       └── retrieval_model.py   Cross-RAG融合ヘッド（公式ChronosBoltを継承しcross/self attention追加）
│
├── retrieval/
│   └── retriever.py    faiss top-k検索器。min-max+metric正規化で検索キー生成、gatherで生窓返却
│
├── data/
│   ├── dataset.py      CSV読込・Informer分割・チャネル標準化・窓スライス(TSData)
│   └── loaders.py      TSData→DataLoader、検索DB(Retriever)構築のグルー
│
└── utils/
    ├── metrics.py      MAE / MSE / RMSE
    └── tools.py        device選択 / median抽出 / EarlyStopping / 結果保存
```

各ファイルの要点：

- **train.py** … `get_method(args)` で手法を取得し、`method.needs_training` が False（`none`・raf naive）なら学習をスキップ。True なら `compute_loss` で学習し `--output-dir/<method>_<dataset>/best.pth` を保存。
- **evaluate.py** … `method.predict` を test 全バッチに適用し、標準化空間で MSE/MAE を計算して `result.txt` に追記。
- **methods/base.py** … `Method` 抽象クラス。`build_model` / `compute_loss` / `predict` と `needs_training` / `needs_retrieval` を定義。train/evaluate はこのIFしか触らない。
- **methods/chronos_base.py** … backbone を凍結して `model(context)` の分位点中央値を点予測に。
- **methods/raf.py** … `build_raf_context` で top-k 検索窓をオフセット整合して連結（context長を超えないよう k を自動cap）。advanced のみ backbone を学習。
- **methods/cross_raf.py** … `INPUT_LEN`/`LAMBDA` を設定して融合ヘッド付きモデルを構築、全融合モジュールを明示初期化（新transformers の未初期化NaN回避）、backbone凍結・融合のみ学習。
- **models/chronos/retrieval_model.py** … 公式 `ChronosBoltModelForForecasting` を継承し、`encode`/`decode` を再利用しつつ cross-attention（query↔検索）と self-attention（検索要約）で融合。
- **retrieval/retriever.py** … 検索キーは min-max → metric正規化（cosine=L2正規化 / correlation=平均除去+L2 / euclidean=min-maxのみ）。`gather` が返す実データは**正規化前の生の標準化窓**。
- **data/dataset.py** … `get_borders` で分割境界、`TSData` で標準化と `make_windows`（`[context|future]` 窓）。
- **_bootstrap.py** … `KMP_DUPLICATE_LIB_OK=TRUE` と OMP スレッド数1を設定。`OMP_NUM_THREADS` で上書き可。

---

## 仕組み・設計メモ

### 手法非依存の共通インターフェース
`train.py` / `evaluate.py` は `methods/base.py` の `Method` IF（`build_model` / `compute_loss` / `predict`）だけを呼ぶ。
手法追加は `methods/` に1ファイル作り `registry.py` に1行足すだけ。

### 検索結果の使い方（3手法の違い）
| method | 使い方 | コード |
|---|---|---|
| none | 使わない | — |
| raf | 入力に整合連結 `[s_{k-1}…s_0 ∥ ctx]` | `methods/raf.py: build_raf_context` |
| cross_raf | cross/self attention で融合 | `models/chronos/retrieval_model.py` |

### 検索の正規化（faiss）
faiss の Flat index は「生の内積」か「生のL2」しか計算しないため、metric用の正規化で類似度を埋め込む：
- `cosine` … min-max → L2正規化 → `IndexFlatIP`
- `euclidean` … min-max → `IndexFlatL2`
- `correlation` … min-max → 平均除去 → L2正規化 → `IndexFlatIP`（＝ピアソン相関）

CUDA環境では faiss GPU index を自動使用。検索の正規化は**検索キー（一時表現）にだけ**掛かり、取り出される窓データは生のまま。

### faiss × torch の OpenMP（macOS）
faiss-cpu と torch がそれぞれ libomp を読み込み二重ロードでセグフォルトする問題を、
`_bootstrap.py`（各エントリ先頭でimport）が `KMP_DUPLICATE_LIB_OK=TRUE` ＋ OMPスレッド数1 で回避。
Linux等でマルチスレッドにしたい場合は `OMP_NUM_THREADS` を明示すれば上書きされる。

### 学習で保存されるもの（チェックポイント）
保存先 `{--output-dir}/{method}_{dataset}/`：

| ファイル | 中身 |
|---|---|
| `model_step{N}.pth` / `best.pth` | `{"state_dict", "optimizer", "step", "args"}` の辞書（自己記述・再開可能）|
| `args.json` | 学習時のハイパラ（人間可読の記録）|

- `evaluate.py --checkpoint` でロードすると **`trained with: ...` で学習設定を表示**。
  `chronos_model / seq_len / pred_len / augment_mode / method` が評価時と食い違うと警告する。
- **`--resume` は厳密な再開**：重み＋optimizer(Adamモーメント)＋step数を復元して途中から正確に続行
  （LRは定数なのでスケジューラ状態は不要）。
- 旧形式（生の `state_dict` のみ）の `.pth` も後方互換で読める（その場合は重みのみ復元と警告）。
- 注意：`best.pth` は検証選択ではなく最終モデル。

### 制約
- backbone は **Chronos-Bolt系のみ**（Moirai/TimesFM等は未対応。`models/` にラッパー追加が必要）。
- `pred_len` は backbone の prediction_length（Boltは全サイズ64）以下、**cross_raf は ==64 必須**。
- 予測・正解はともに標準化空間で比較するため、3手法を公平に比較できる。

### 既定ハイパラの根拠
学習系（`lr=3e-4` / `train_steps=10000` / `batch=256` / `weight_decay=0.01` / `top_k=15` / dropout=0.2）は
Cross-RAG 論文 Table A.2 準拠。`raf --raf-mode advanced` は backbone 全体を微調整するので `--lr` を 1e-5〜1e-4 に下げるのを推奨。
