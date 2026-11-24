# Retrieval-Augmented Forecasting with Chronos-2

凍結した Chronos-2 に，過去の窓（事例）を渡して予測する．論旨と結果は `docs/raf-note.html`．

提案手法は 4 つの組み合わせで，`--select attn --heads <選択ヘッド>` で走る（`--select` と `--heads` 以外の既定は提案手法の値）．選択ヘッドはデータごとに `heads.py` で val から選ぶ（ETTh1 は `L12H1,L11H10,L10H12`）．

- (A) 事例を共変量の行として並列に置き，group attention で読ませる（`--fusion parallel`）
- (B) 最初の予測パッチの列だけ，クエリから事例を隠す（`--hide P1`）
- (C) 選択ヘッド 3 つの attention weight で事例を選ぶ（`--select attn --heads ...`）
- (D) 候補ごとに key を前計算し，クエリごとに train の窓から引いた候補 2048 件から 256 件を選ぶ（`--head-weights precompute --candidates 2048 --top-k 256`）

## 構成

| ファイル | 役割 |
|---|---|
| `codes/dataset.py` | parquet の読み込み，train/val/test の分割，窓の切り出し |
| `codes/chronos2.py` | Chronos-2 の forward．並列と連結，隠す列，ヘッドの query・key と重みの取り出し |
| `codes/selection.py` | 候補の引き方と選び方（ランダム，照合 5 種，ヘッドの重み，続きを隠して流した q・k・v を使う点数，状態の距離） |
| `codes/run.py` | 入口．予測と正解を `results/<名前>.parquet` に縦長で保存（1 行＝チャネル×クエリ×予測の何点目 `h`．列は `date` `true` `q0.1`〜`q0.9` など） |
| `codes/heads.py` | `--select attn` と `--select state` で使うヘッド（層）を，点数ごとに val（既定）か train の末尾のクエリで選び直す（test は使わない） |
| `codes/evaluate.py` | MSE・MAE・分位点損失（QL）．各チャネルを train の標準偏差で割って 7 チャネルで平均する．`eMSE` などは評価区間の正解の標準偏差で割った補助指標．`--compare` で提案手法などとの差を標準誤差つきで，`--picks` で選んだ事例を oracle と比べる |
| `codes/contrib.py` | 選んだ事例の寄与 Σ s(v − v₀)W_O を前項と変化項に分け，クエリが自分の続きを知ったときの変化 (v₀* − v₀)W_O と比べる（val，前計算．予測対象は分析にだけ使う） |

## 準備

```bash
pip3 install "chronos-forecasting>=2.0" pandas pyarrow einops
```

`Datasets/_parquet/<dataset>.parquet`（`date` 列＋各チャネル）を置く．

## 実行

```bash
python3 codes/run.py                          # 参照なし
python3 codes/run.py --select attn --heads L12H1,L11H10,L10H12              # 提案手法（ETTh1）
python3 codes/run.py --select attn --heads L12H1,L11H10,L10H12 --hide ""    # 提案から (B) を外す
python3 codes/run.py --select random          # 同じ候補・件数・隠し方でランダム
python3 codes/run.py --select ctx=ctx --top-k 8 --fusion concat
python3 codes/run.py --select attn --heads L12H1,L11H10,L10H12 --eval-split test   # 値を val で決めてから
python3 codes/evaluate.py results/*.parquet
```

`--select` 以外の既定は提案手法の値なので，ほかの選び方も同じ候補・件数・隠し方で比べられる．参照なしでは，履歴・予測の長さ・起点の間隔だけが効く．

| 引数 | 既定 | 意味 |
|---|---|---|
| `--select` | `none` | `none`：参照なし．`random`：候補の最初の K 件．照合：`ctx=ctx` `fc=fc` `fc=fut` `fut=fc` `fut=fut`．`attn`：選択ヘッドの attention の重み．`state`：続きを隠して流した状態がクエリに近い順 |
| `--fusion` | `parallel` | `parallel`：事例を共変量の行に．`concat`：事例を履歴の前に時間方向につなぎ，あいだに欠測を 1 パッチ空ける |
| `--top-k` | `256` | 事例の数 K．連結は履歴 512 点なら 16 件，2032 点なら 13 件まで |
| `--candidates` | `2048` | クエリごとにランダムに引く候補の数．0 は使える全窓．どの選び方もこの中から選ぶ |
| `--hide` | `P1` | クエリが事例を読まない列．`P1`〜`P4` は予測パッチ，`C1` は文脈の最後のパッチ（`C2` はその前），`REG`．例：`P1,C1`．`""` で隠さない．並列のときだけで，連結では隠さない |
| `--metric` | `l2` | 照合の距離．`l2`（ユークリッド距離），`cosine`（1 − コサイン類似度），`corr`（1 − 相関係数） |
| `--match-norm` | `context` | `context`：クエリと窓をそれぞれ自分の文脈で標準化してから測る．`raw`：生の値で測る |
| `--heads` | なし（`attn` では必須） | `attn` で使う選択ヘッド（第何層の何番目か，1 から数える）．`heads.py` の出力をデータごとに渡す |
| `--head-weights` | `precompute` | `precompute`：各行を単独で通して key を前計算する．`exact`：クエリと候補をまとめて通す（候補 2048 件はメモリに載らない） |
| `--select-hide` | なし | `exact` で選ぶ 1 回目の推論にも `--hide` をかける |
| `--value-score` | `none` | `attn` で，候補を続きを隠しても通し，q，k，v を「全体・前項・変化項」で使う（`precompute` のみ）．q₀，k₀，v₀ はクエリ，q，k，v は候補，~ は候補の続きを隠したもの．`q` `k`：‖q − q₀‖ の小さい順，`qpre` `kpre`：‖q~ − q₀‖ の小さい順，`qchange` `kchange`：‖q − q~‖ の大きい順（`_s` を付けると小さい順）．`v`：‖(v − v₀)W_O‖ の大きい順，`vpre`：‖(v~ − v₀)W_O‖ の小さい順，`vchange`：‖(v − v~)W_O‖ の大きい順（`vchange_s` は小さい順）．`qkpre`：q₀·k~ の softmax，`qkchange`：q₀·(k − k~) の softmax．`qkv` `qkvpre` `qkvchange`：softmax(q₀·k) ‖(v − v₀)W_O‖ などを列で足したもの．`avpre`：softmax(q₀·k) ÷ ‖(v~ − v₀)W_O‖，`avchpre`：それに ‖(v − v~)W_O‖ を掛けたもの．ノルムは予測パッチ 4 列とヘッドで足す（式は `selection.py` の `head_terms`）．次の 8 つは `heads.py` では測らない（式は `masked_score`）．`vscale`：qk に 2‖v₀W_O‖‖vW_O‖ ÷ (‖v₀W_O‖² + ‖vW_O‖²) の列平均をヘッドごとに掛けたもの．`cosvscale`：qk の代わりに cos(q₀, k) の列平均を使って同じように掛けたもの．`vstab`：qk を 1 + Σ‖(v − v₀)W_O‖ ÷ Σ‖v₀W_O‖ で割ったもの．`vnorm`：qk を Σ‖vW_O‖ で割ったもの（Σ は予測パッチ 4 列とヘッドの和）．`qkvo`：softmax(q₀·k)(v − v₀)W_O をヘッドで足してから L2 ノルムを取り，列で足したもの（W_O を横に並べたグラム行列のコレスキー因子を使う）．`qkbi`：qk に，候補の q で取った 2 行の softmax σ(q·k₀ − q·k)（候補の行がクエリを読む量）の列平均を掛けたもの．`jac` `jac_s`：softmax(q₀·k)(v − v₀)W_O が予測 64 時点を動かす量 ‖J …‖ の大きい順と小さい順．J はクエリを単独で流した予測（中央値）の，各層のグループ注意の出力についてのヤコビアンで，`--hide` の列は 0 |
| `--state` | なし（`state` では必須） | `out`：`--layers` の層の group attention の出力（全ヘッド）．`in`：同じ層の group attention の入力（layer norm の後）．`final`：エンコーダの最終状態（予測の直前）．どれも予測パッチ 4 列で，距離はクエリとの差の L2 ノルムを層と列で足したもの |
| `--layers` | なし | `--state out` と `in` で使う層（1 から数える），例：`L12,L11,L3` |
| `--select-only` | なし | 事例を選ぶだけで予測しない．`_selected.parquet` と `_args.json` だけを書く |
| `--history` | `512` | クエリの履歴の長さ．先行研究（TS-RAG，Cross-RAG など）と同じ．長い履歴は 8192（上限），10-28 の報告書は 2032 |
| `--seq-len` / `--pred-len` | `512` / `64` | 事例の文脈の長さ（履歴より長くできない）/ 予測の長さ．事例の文脈はクエリの履歴と同じ長さにする（RAFT・TS-RAG と同じ）．K と候補の数は履歴 2032 点で選んだ値のまま |
| `--eval-stride` | `17` | 予測の起点の間隔．24 と 168 と公約数を持たない |
| `--db-stride` | `1` | train の窓を切る間隔（1 は全窓） |
| `--seed` | `0` | 候補を引く乱数 |
| `--eval-split` | `val` | `val` / `test`．ハイパーパラメータは val で決めてから test を出す |
| `--overwrite` | なし | 同名の結果があると止まる．上書きするときに付ける |

照合の名前は「クエリの側＝窓の側」で，`ctx` は文脈（クエリは履歴の最後の `seq_len` 点），`fut` は未来，`fc` は参照なしの予測（クエリは履歴から，窓は自分の文脈 `seq_len` 点だけから）．`fut` をクエリの側に使う `fut=fc` と `fut=fut` は予測対象を読むカンニング（oracle）で，手法ではない．

候補は，予測の起点より 96 点以上前に終わる train の窓から引く．乱数は seed，チャネル名，クエリの番号から作るので，選び方を変えても同じクエリには同じ候補が渡る．
`--head-weights precompute` と `--select state` で使う train の窓のベクトルは，`results/cache/<データ>_<チャネル>_s<文脈>_p<予測>_db<間隔>_{fut,mask}_` に続けて，ヘッドなら `L<層>H<ヘッド>_{q,k,v}.npy`，層なら `L<層>_{in,out}.npy`，最終状態なら `final.npy` の名前で保存し，同じ設定の次の実行から読み出す（1 ヘッド 1 チャネルあたり約 8 MB，1 層は約 99 MB）．データの parquet を差し替えたら消す．
`_selected.parquet` には選んだ窓（`db_index` は train の窓の番号で，開始位置は `db_index × db-stride`）と，その選び方での点数が入る．

結果の名前は `<データセット>_<分割>_<選び方>` に，既定と違う値が付く（例：提案手法は `ETTh1_val_attn`，隠さないと `ETTh1_val_attn_nohide`）．既定は `run.py` 冒頭の `DEFAULTS` にある．

## 選び方を比べる

```bash
python3 codes/run.py --select attn --heads L12H1,L11H10,L10H12 --value-score avpre   # 予測を流す
python3 codes/evaluate.py --compare results/ETTh1_val_none results/ETTh1_val_attn_L12H1+L11H10+L10H12 \
    results/ETTh1_val_attn_L12H1+L11H10+L10H12_vavpre --out results/ETTh1_val_select_eval.json
python3 codes/run.py --select state --state final --select-only   # 選ぶだけ
python3 codes/evaluate.py --picks results/ETTh1_val_fut-fut results/ETTh1_val_state_final_selonly
```

`--compare` は，参照なしに対する MSE・MAE の変化率，基準の run との差（起点をチャネルごとに 1000 回復元抽出した標準誤差つき），動きと必要な修正（予測対象 − 参照なし）の符号が同じ時点の割合，動きの大きさ，チャネルごとの基準との MSE の差，MSE が基準より小さい起点の割合を出す．どの run も全起点をそろえて渡す．`--picks` は，予測を流さずに，選んだ事例が oracle（`--select fut=fut`）の 256 件と重なる割合と，事例の続きと予測対象の RMSE（`dist`，それぞれ自分の文脈で標準化），選んだ事例どうしの RMSE の平均（`spread`），事例の続きの平均と予測対象の RMSE（`center`）を出す．

## 事例の寄与を測る

```bash
python3 codes/contrib.py --heads L12H1,L11H10,L10H12 --runs results/ETTh1_val_attn_L12H1+L11H10+L10H12 \
    results/ETTh1_val_attn_L12H1+L11H10+L10H12_vavpre results/ETTh1_val_random --out results/ETTh1_val_contrib.json
```

`--runs` の `_selected.parquet` が選んだ 256 件 S について，`--heads` のヘッドと予測パッチ 4 列で，重み s をクエリ自身と S の上の softmax(q₀·k) にとり，寄与 Δ = Σ s(v − v₀)W_O を前項の分 Σ s(v~ − v₀)W_O と変化項の分 Σ s(v − v~)W_O に分ける．クエリを予測対象つきで単独に流した v₀*（* は予測対象を入れて流した値）から u₀ = (v₀* − v₀)W_O を作り，Δ・前項の分・変化項の分と u₀ の cos（`cos` `cos_pre` `cos_chg`），cos(Δ, u₀) を前項の分と変化項の分に足し算で分けた ⟨Δ_pre, u₀⟩/(‖Δ‖‖u₀‖) と ⟨Δ_chg, u₀⟩/(‖Δ‖‖u₀‖)（`a_pre` `a_chg`），‖Δ_pre‖/‖u₀‖ と ‖Δ_chg‖/‖u₀‖（`n_pre` `n_chg`），Δ の u₀ 方向の成分 ⟨Δ, u₀⟩/‖u₀‖²（`gain`）と直交する成分の長さ ÷ ‖u₀‖（`off`），選んだ事例の前項と変化項の大きさの平均（`dpre` `dchg`），候補全体での前項と変化項の大きさの順位相関（`corr`）を出す．`--ref`（既定は最初の run）との差に，起点をチャネルごとに復元抽出した標準誤差を付ける．値はどれも前計算（行を 1 本ずつ流した値）で，キャッシュの k と v を使う．

## ヘッドを選び直す

```bash
python3 codes/heads.py --head-weights exact                            # 提案手法のヘッド（11-18 の報告書）
python3 codes/heads.py --head-weights precompute                       # 点数ごと（10〜12 層から選ぶ）
```

各チャネルの val に均等に散らばる 60 個の起点（`--split train` なら train の最後の 60 個，履歴は train の中．どちらも `--eval-stride` おき）をクエリにし，それぞれ train の窓を 512 件ランダムに候補にする．ヘッド（層）ごとに，候補の点数と，候補の未来がクエリの予測対象にどれだけ近いか（それぞれ自分の文脈で標準化した RMSE）の順位相関をとり，クエリで平均する．`--layers`（既定は 10〜12 層）の中で相関の高い `--top 3` 個を出し，相関の表を `results/heads_<データセット>_<分割>_<重み>_h<履歴>.json` に保存する．
`--head-weights` は必須．`exact` はクエリと候補を 1 回の forward に一緒に入れた attention weight だけを測る．`precompute` は行ごとに単独で通し，attention weight（`attn`，q₀·k の softmax），`--value-score` のうち `qkvo` `qkbi` `jac` `jac_s` `vscale` `vstab` `vnorm` `cosvscale` を除くすべて，層の出力と入力の距離（`out` `in`），最終状態の距離（`final`）をまとめて測る．2 つの重みは同じヘッドでも相関が大きく違うことがある（ETTh1 の val で L10H12 は exact 0.212，precompute 0.033）．層は 10〜12 に絞る．ETTh1 の val で 1〜12 層から選ぶと，qk は L12H1，L12H7，L2H9 になり，参照なしに対する MSE の変化率は −3.13% だった（10〜12 層の L12H1，L12H7，L12H6 は −4.95%）．相関は全ヘッドについて計算して JSON に残るので，範囲は後から変えられる．

`python3 codes/chronos2.py` で，並列の計算が Chronos-2 本来の forward と一致すること，全部の列を隠すと参照なしと一致することを確認できる．
