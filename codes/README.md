# Retrieval-Augmented Forecasting with Chronos-2

凍結した Chronos-2 に，過去の窓（参照事例）を**共変量の行として並列に**渡して予測する．
着目点は 2 つ：**どう融合するか**（並列）と，**どう選ぶか**（全クエリ共通の many-shot セット）．
論旨と結果は `docs/raf-note.html`．旧実験の記録は git 履歴（`c66158c` 以前）と `results/_archive/` にある．

## 構成

| ファイル | 役割 |
|---|---|
| `codes/dataset.py` | parquet 読み込み，train/val/test 分割，窓の切り出し |
| `codes/chronos2.py` | Chronos-2 の forward．並列（事例を共変量の行に）と連結（事例を時間軸の前に）．事例 1 件 11 トークンで計算するので many-shot が回る |
| `codes/selection.py` | 事例の選び方．task-level：`random` / `kmeans`．instance-level：`l2` / `oracle` / `recent` / `placebo` |
| `codes/run.py` | 入口．予測と正解を `results/<名前>.parquet` に保存 |
| `codes/evaluate.py` | チャネル正規化 MSE と分位点損失（QL） |

## 準備

```bash
pip3 install "chronos-forecasting>=2.0" pandas pyarrow scikit-learn einops
```

`Datasets/_parquet/<dataset>.parquet`（`date` 列＋各チャネル）を置く．

## 実行

```bash
python3 codes/run.py --select none                                  # 参照なし
python3 codes/run.py --select random --top-k 128 --seed 0           # task-level・many-shot
python3 codes/run.py --select l2 --top-k 8                          # instance-level（RAF の基準線）
python3 codes/run.py --select l2 --top-k 8 --fusion concat          # 同じ事例を連結で
python3 codes/run.py --select random --top-k 32 --ablation shuffle-future
python3 codes/evaluate.py results/*.parquet
```

| 引数 | 既定 | 意味 |
|---|---|---|
| `--fusion` | `parallel` | `parallel`：事例を共変量の行に．`concat`：事例を履歴の前に連結（qmean 区切り 16 点） |
| `--select` | `none` | 選び方（上の表） |
| `--top-k` | `8` | 事例の数．連結は履歴 2032 なら 35 件まで |
| `--history` | `2032` | クエリの履歴長 |
| `--seq-len` / `--pred-len` | `96` / `64` | 事例の文脈長 / 予測長 |
| `--db-stride` | `8` | 事例の候補となる窓の間隔 |
| `--ablation` | `none` | `shuffle-future`：事例どうしで未来を入れ替える．`truth-future`：未来を正解に差し替える（診断用） |
| `--eval-split` | `test` | `val` / `test`．ハイパーパラメータは val で決めてから test を出す |

task-level の事例は train の窓から選ぶ．instance-level の事例は，クエリの文脈が始まる前に終わる全ての窓から選ぶ．
`oracle` と `truth-future` は正解を使う診断用で，手法ではない．

`python3 codes/chronos2.py` で，並列の計算が Chronos-2 本来の forward と一致することを確認できる．
