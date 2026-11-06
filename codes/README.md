# Retrieval-Augmented Forecasting with Chronos-2

凍結した Chronos-2 に，過去の窓（事例）を渡して予測する．論旨と結果は `docs/raf-note.html`．

提案手法は 3 つの組み合わせで，`--select attn` だけで走る（`--select` 以外の既定は提案手法の値）．

- (A) 事例を共変量の行として並列に置き，group attention で読ませる（`--fusion parallel`）
- (B) 最初の予測パッチの列だけ，クエリから事例を隠す（`--hide P1`）
- (C) クエリごとに train の窓から候補を 2048 件引き，近さヘッド 3 つの重みで 256 件に絞る．key は前計算する（`--select attn --candidates 2048 --top-k 256`）

## 構成

| ファイル | 役割 |
|---|---|
| `codes/dataset.py` | parquet の読み込み，train/val/test の分割，窓の切り出し |
| `codes/chronos2.py` | Chronos-2 の forward．並列と連結，隠す列，ヘッドの query・key と重みの取り出し |
| `codes/selection.py` | 候補の引き方と選び方（ランダム，照合 5 種，ヘッドの重み） |
| `codes/run.py` | 入口．予測と正解を `results/<名前>.parquet` に縦長で保存（1 行＝チャネル×クエリ×予測の何点目 `h`．列は `date` `true` `q0.1`〜`q0.9` など） |
| `codes/heads.py` | `--select attn` で使う近さヘッドを，val（既定）か train の末尾のクエリで選び直す（test は使わない） |
| `codes/evaluate.py` | MSE・MAE・分位点損失（QL）．各チャネルを train の標準偏差で割って 7 チャネルで平均する．`eMSE` などは評価区間の正解の標準偏差で割った補助指標 |

## 準備

```bash
pip3 install "chronos-forecasting>=2.0" pandas pyarrow einops
```

`Datasets/_parquet/<dataset>.parquet`（`date` 列＋各チャネル）を置く．

## 実行

```bash
python3 codes/run.py                          # 参照なし
python3 codes/run.py --select attn            # 提案手法
python3 codes/run.py --select attn --hide ""  # 提案から (B) を外す
python3 codes/run.py --select random          # 同じ候補・件数・隠し方でランダム
python3 codes/run.py --select ctx=ctx --top-k 8 --fusion concat
python3 codes/run.py --select attn --eval-split test   # 値を val で決めてから
python3 codes/evaluate.py results/*.parquet
```

`--select` 以外の既定は提案手法の値なので，ほかの選び方も同じ候補・件数・隠し方で比べられる．参照なしでは，履歴・予測の長さ・起点の間隔だけが効く．

| 引数 | 既定 | 意味 |
|---|---|---|
| `--select` | `none` | `none`：参照なし．`random`：候補の最初の K 件．照合：`ctx=ctx` `fc=fc` `fc=fut` `fut=fc` `fut=fut`．`attn`：近さヘッドの attention の重み |
| `--fusion` | `parallel` | `parallel`：事例を共変量の行に．`concat`：事例を履歴の前に時間方向につなぎ，あいだに欠測を 1 パッチ空ける |
| `--top-k` | `256` | 事例の数 K．連結は履歴 512 点なら 16 件，2032 点なら 13 件まで |
| `--candidates` | `2048` | クエリごとにランダムに引く候補の数．0 は使える全窓．どの選び方もこの中から選ぶ |
| `--hide` | `P1` | クエリが事例を読まない列．`P1`〜`P4` は予測パッチ，`C1` は文脈の最後のパッチ（`C2` はその前），`REG`．例：`P1,C1`．`""` で隠さない．並列のときだけで，連結では隠さない |
| `--metric` | `l2` | 照合の距離．`l2`（ユークリッド距離），`cosine`（1 − コサイン類似度），`corr`（1 − 相関係数） |
| `--match-norm` | `context` | `context`：クエリと窓をそれぞれ自分の文脈で標準化してから測る．`raw`：生の値で測る |
| `--heads` | `L12H1,L12H7,L11H10` | `attn` で使うヘッド（第何層の何番目か，1 から数える） |
| `--head-weights` | `precompute` | `precompute`：各行を単独で通して key を前計算する．`exact`：クエリと候補をまとめて通す（候補 2048 件はメモリに載らない） |
| `--select-hide` | なし | `exact` で選ぶ 1 回目の推論にも `--hide` をかける |
| `--history` | `512` | クエリの履歴の長さ．先行研究（TS-RAG，Cross-RAG など）と同じ．長い履歴は 8192（上限），報告書は 2032 |
| `--seq-len` / `--pred-len` | `512` / `64` | 事例の文脈の長さ（履歴より長くできない）/ 予測の長さ．事例の文脈はクエリの履歴と同じ長さにする（RAFT・TS-RAG と同じ）．K と候補の数は履歴 2032 点で選んだ値のまま |
| `--eval-stride` | `17` | 予測の起点の間隔．24 と 168 と公約数を持たない |
| `--db-stride` | `1` | train の窓を切る間隔（1 は全窓） |
| `--seed` | `0` | 候補を引く乱数 |
| `--eval-split` | `val` | `val` / `test`．ハイパーパラメータは val で決めてから test を出す |
| `--overwrite` | なし | 同名の結果があると止まる．上書きするときに付ける |

照合の名前は「クエリの側＝窓の側」で，`ctx` は文脈（クエリは履歴の最後の `seq_len` 点），`fut` は未来，`fc` は参照なしの予測（クエリは履歴から，窓は自分の文脈 `seq_len` 点だけから）．`fut` をクエリの側に使う `fut=fc` と `fut=fut` は正解を使う診断で，手法ではない．

候補は，予測の起点より 96 点以上前に終わる train の窓から引く．乱数は seed，チャネル名，クエリの番号から作るので，選び方を変えても同じクエリには同じ候補が渡る．
`_selected.parquet` には選んだ窓（`db_index` は train の窓の番号で，開始位置は `db_index × db-stride`）と，その選び方での点数が入る．

結果の名前は `<データセット>_<分割>_<選び方>` に，既定と違う値が付く（例：提案手法は `ETTh1_val_attn`，隠さないと `ETTh1_val_attn_nohide`）．既定は `run.py` 冒頭の `DEFAULTS` にある．

## ヘッドを選び直す

```bash
python3 codes/heads.py                                   # ETTh1，val，正確な重み
python3 codes/heads.py --dataset ETTh2 --split train --head-weights precompute
```

各チャネルの val に均等に散らばる 60 個の起点（`--split train` なら train の最後の 60 個，履歴は train の中．どちらも `--eval-stride` おき）をクエリにし，それぞれ train の窓を 512 件ランダムに候補にする．ヘッドごとに，クエリが候補に置く重み（予測の 4 列の平均）と，候補の未来がクエリの正解にどれだけ近いか（それぞれ自分の文脈で標準化した RMSE）の順位相関をとり，クエリで平均する．`--layers 10-12` の中で相関の高い `--top 3` 個を `--heads` の形で出し，144 ヘッドの相関を `results/heads_<データセット>_<分割>_<重み>_h<履歴>.json` に保存する．
既定の `--heads`（L12H1，L12H7，L11H10）は，ETTh1，train の末尾，履歴 2032 点，正確な重みで選んだもの．

`python3 codes/chronos2.py` で，並列の計算が Chronos-2 本来の forward と一致すること，全部の列を隠すと参照なしと一致することを確認できる．
