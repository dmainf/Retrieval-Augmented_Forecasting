# RAF Playground — 3手法を1つのCLIで切り替える検索拡張時系列予測

Chronos-Bolt をバックボーンに、以下の3手法を `--method` で切り替えて学習・評価できる軽量フレームワーク。
`cross-rag/` のモデル実装を流用しつつ、データローダ・検索は依存の軽い自前実装にしている。

| `--method` | 内容 | 検索 | 融合 | 学習 |
|---|---|---|---|---|
| `none` | 素の Chronos（zero-shot） | なし | なし | なし |
| `raf` | コンテキスト拡張 RAF (Tire et al. 2026) | top-1 | 入力連結 | naive=凍結 / advanced=微調整 |
| `cross_raf` | Cross-RAG (Lee et al. 2026) | top-k | cross + self attention | 融合モジュールのみ |

## セットアップ

```bash
pip3 install -r requirements.txt          # 失敗時は --break-system-packages
```

バックボーンは初回に Hugging Face から `amazon/chronos-bolt-small` を取得する。

## データ

`--root-path` 直下に `date` 列＋変数列を持つ標準ベンチCSV（ETTh1.csv, weather.csv など）を置く。
各変数列を独立した1変量系列として扱い、Informer流の train/val/test 分割・チャネル標準化を行う。
評価・学習・検索DBは全てこの分割から構築する（test = クエリ、train = 検索DB）。

本リポジトリ同梱データ（`./Datasets/`）：
- `./Datasets/ETT-small/`  … `ETTh1.csv` `ETTh2.csv` `ETTm1.csv` `ETTm2.csv`（`--dataset ETTh1` 等で分割が切替わる）
- `./Datasets/weather/weather.csv`、`./Datasets/exchange_rate/exchange_rate.csv`、`./Datasets/electricity/electricity.csv`
  （これらは custom 扱い。`--dataset custom` または各データ名を指定）

## 使い方

### 1) 素の Chronos（学習不要）
```bash
python3 evaluate.py --method none \
  --dataset ETTh1 --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv \
  --seq-len 512 --pred-len 64
```

### 2) RAF (naive＝学習不要)
```bash
python3 evaluate.py --method raf --raf-mode naive \
  --dataset ETTh1 --root-path ./Datasets/ETT-small/ --data-path ETTh1.csv
```

### 2') RAF (advanced＝backbone微調整)
```bash
python3 train.py    --method raf --raf-mode advanced --dataset ETTh1 ... --train-steps 20000
python3 evaluate.py --method raf --raf-mode advanced --dataset ETTh1 ... \
  --checkpoint ./checkpoints/raf_ETTh1/best.pth
```

### 3) Cross-RAF（融合モジュール学習 → 評価）
```bash
python3 train.py    --method cross_raf --dataset ETTh1 ... --top-k 15 --train-steps 20000
python3 evaluate.py --method cross_raf --dataset ETTh1 ... --top-k 15 \
  --checkpoint ./checkpoints/cross_raf_ETTh1/best.pth
```

## 主なオプション（`config.py`）

```
--method {none,raf,cross_raf}     手法切替
--raf-mode {naive,advanced}       raf時のみ
--dataset / --root-path / --data-path / --features {M,S} / --target
--seq-len --pred-len              pred-len は backbone の prediction_length(=64) 以下
                                  （cross_raf は ==64 が必須）
--top-k --retrieval-metric {cosine,euclidean,correlation}   # faiss検索
--chronos-model amazon/chronos-bolt-small
--output-dir --batch-size --gpu --seed
# train.py 追加: --epochs --train-steps --lr --weight-decay --train-stride --save-freq --resume
# evaluate.py 追加: --checkpoint --eval-stride --result-file
```

## ディレクトリ構成

```
raf/
├── train.py / evaluate.py / config.py
├── methods/        # registry + base + none/raf/cross_raf（raf.pyに連結ロジック）
├── models/chronos/ # Chronos-Bolt backbone（公式chronos + cross_raf融合ヘッド）
├── retrieval/      # retriever(faiss top-k)
├── data/           # CSV窓スライス・分割・DataLoader
└── utils/          # metrics / tools
```

## 設計メモ
- 各手法は `methods/base.py` の共通IF（`build_model` / `compute_loss` / `predict`）を実装し、
  `train.py` / `evaluate.py` は手法非依存。手法追加は `methods/` に1ファイル＋registry登録で完結。
- 検索は **faiss** で X 空間（min-max正規化）の top-k。`--retrieval-metric` で類似度を選択：
  - `cosine`      … L2正規化 + `IndexFlatIP`
  - `euclidean`   … `IndexFlatL2`
  - `correlation` … 平均除去 + L2正規化 + `IndexFlatIP`（ピアソン相関）
  - CUDA環境では faiss GPU index を自動使用。
- 予測・正解はともに標準化空間で比較し、3手法を公平に比較できる。

## faiss × torch の OpenMP について（macOS）
faiss-cpu と torch はそれぞれ独自の OpenMP(libomp) を読み込むため、Mac では二重ロードで
セグフォルトすることがある。`_bootstrap.py` を各エントリの最初に import して
`KMP_DUPLICATE_LIB_OK=TRUE` と OMP スレッド数 1 を設定し回避している（検証済み）。
Linux 等でマルチスレッドにしたい場合は `OMP_NUM_THREADS` を明示的に設定すれば上書きできる。
