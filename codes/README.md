# Retrieval-Augmented Forecasting with Chronos-2

凍結した Chronos-2 に，過去の窓（参照事例）を**共変量の行として並列に**渡して予測する．
着目点は 2 つ：**どう融合するか**（並列）と，**どう選ぶか**（全クエリ共通の many-shot セット）．
論旨と結果は `docs/raf-note.html`．旧実験の記録は git 履歴（`c66158c` 以前）と `results/_archive/` にある．

## 構成

| ファイル | 役割 |
|---|---|
| `codes/dataset.py` | parquet 読み込み，train/val/test 分割，窓の切り出し |
| `codes/chronos2.py` | Chronos-2 の forward．並列（事例を共変量の行に）と連結（事例を時間軸の前に）．事例 1 件 11 トークンで計算するので many-shot が回る |
| `codes/selection.py` | 事例の選び方．task-level：`random` / `clg` / `pclg`（Parallel-CLG，提案手法）．instance-level：`l2` / `oracle` / `truth` |
| `codes/run.py` | 入口．予測と正解を `results/<名前>.parquet` に縦長で保存（1 行＝チャネル×クエリ×予測の何点目 `h`．列は `date` `true` `q0.1`〜`q0.9` など） |
| `codes/clg.py` | CLG と Parallel-CLG の潜在の学習と勾配の軌跡．`--select clg|pclg` の前に実行する．選び方は pclg がコサイン，clg が L2（平均に合わせる）が既定で，`--match` で変えられる．`--inst-k` で共通セットに文脈照合を足す（ハイブリッド） |
| `codes/evaluate.py` | MSE・MAE・分位点損失（QL）．主な指標は各チャネルを train の標準偏差で標準化した値（論文の慣例）．`eMSE` などは評価区間の正解の標準偏差で割った補助指標．以前の横長の表も読める |

## 準備

```bash
pip3 install "chronos-forecasting>=2.0" pandas pyarrow einops
```

`Datasets/_parquet/<dataset>.parquet`（`date` 列＋各チャネル）を置く．

## 実行

```bash
python3 codes/run.py --select none                                  # 参照なし
python3 codes/run.py --select random --top-k 128 --seed 0           # task-level・many-shot
python3 codes/run.py --select l2 --top-k 8                          # instance-level（RAF の基準線）
python3 codes/run.py --select l2 --top-k 8 --fusion concat          # 同じ事例を連結で
python3 codes/run.py --select random --top-k 32 --ablation shuffle-future
python3 codes/run.py --select random --top-k 128 --seed 0 --eval-split test   # 値を val で決めてから
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
| `--ablation` | `none` | `shuffle-future`：事例どうしで未来を入れ替える．選び方によらずかかる（`truth` は K 件が同じなので変化しない） |
| `--eval-split` | `val` | `val` / `test`．ハイパーパラメータは val で決めてから，`--eval-split test` で test を出す |
| `--overwrite` | なし | 同名の結果があると止まる．上書きするときに付ける |

どの選び方も，候補は同じ train の窓（`--db-stride` 点おき）．instance-level は，そのうちクエリの文脈が始まる前に終わる窓から選ぶ．
`_selected.parquet` の `db_index` は train の窓の番号で，窓の開始位置は `db_index × db-stride`．
`oracle` と `truth` は正解を使う診断用で，手法ではない．`truth` はクエリ自身の文脈と正解の未来をそのまま K 件渡す（候補からは選ばない）．

ファイル名は `<データセット>_<分割>_<選び方>_k<K>_<融合>_s<seed>` に，既定と違う `--seq-len` `--pred-len` `--history` `--eval-stride` `--db-stride`（`_h1024` `_db16` など）と `--channels`（`_chOT`）が付く．
乱数はチャネルごとに seed とチャネル名から作るので，`--channels` で一部だけ走らせても同じ事例が選ばれる．

ハイパーパラメータはすべて `run.py` 冒頭の定数にまとめてある（引数の既定値，連結の区切り長 `SEP_LEN`，分割比，モデル名，保存する分位点，`TOKEN_BUDGET`）．
連結の区切り（qmean）は整理前に test 上で選んだ値を引き継いでいる．val での確認はまだ．
`--eval-stride 17` と `--db-stride 11` は，1 日（24）と 1 週間（168）の周期と公約数を持たない値．db-stride は val で 5 / 7 / 8 / 11 を比べて決めた（差は誤差の範囲）．

`python3 codes/chronos2.py` で，並列の計算が Chronos-2 本来の forward と一致することを確認できる．
