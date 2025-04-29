CLIPort(https://github.com/cliport)をDockerで動かせるようデバッグし、ファインチューニング用のプログラムを作成しました。
今後も開発を続けるため、変更する可能性があります。変更点は随時ここに記入します。

This repository is a modified version of [CLIport](https://github.com/cliport/cliport), originally released under the Apache License 2.0.  
The original license is preserved in the LICENSE file.


# オリジナルからの変更点
・Dockerfile：デバッグをしました。

・cliport\utils\utils.py：ImageRotatorクラスをデバッグをしました。

・cliport\models\core\clip.py：load_clip関数をデバッグしました。

・cliport\finetune.py：ファインチューニング用プログラムを作成しました。

# 実装手順
### 0.インストール確認
docker desktopやcudaがインストールされていて、アクティブになっているか確認してください。

### 1.Dockerビルド
windowsのPSにて実行しています！

下記のどちらかでビルド
```
docker build -t cliport_ws .
docker build --no-cache -t cliport_ws .
```

### 2.Docker ラン & マウント
```
 docker run -it --rm --gpus all `
>>   -v ${PWD}:/workspace/cliport `
>>   -w /workspace/cliport `
>>   cliport_ws
```

### 3.Demo実装
コンテナの環境変数を追加してください。
```
export CLIPORT_ROOT=/workspace/cliport
```

```
python3 -m cliport.demos n=10 task=stack-block-pyramid-seq-seen-colors mode=test
```
モジュールとして実行してください。

### 4.Test実装
```
python3 -m cliport.eval model_task=multi-language-conditioned \
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=10 \
                       train_demos=1000 \
                       exp_folder=cliport_quickstart \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False \
                       record.save_video=True
```

# 学習方法
### 1.シングルタスク学習
```
python3 -m cliport.demos n=1000 task=stack-block-pyramid-seq-seen-colors mode=train
python3 -m cliport.train train.task=stack-block-pyramid-seq-seen-colors \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=1000 \
                        train.n_steps=201000 \
                        train.exp_folder=exps \
                        dataset.cache=False 
```
### 2.ファインチューニング
```
python3 -m cliport.demos n=1000 task=stack-block-pyramid-seq-seen-colors mode=train
python3 -m cliport.finetune train.task=stack-block-pyramid-seq-seen-colors \
                           train.agent=cliport \
                           train.attn_stream_fusion_type=add \
                           train.trans_stream_fusion_type=conv \
                           train.lang_fusion_type=mult \
                           train.n_demos=100 \
                           train.n_steps=101000 \
                           train.exp_folder=exps \
                           dataset.cache=False \
```
### 3.バリデーション
```
python3 -m cliport.demos n=100 task=stack-block-pyramid-seq-seen-colors mode=val
python3 -m cliport.eval eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps 
```

### 4.テスト
```
python3 -m cliport.demos n=100 task=stack-block-pyramid-seq-seen-colors mode=test
python3 -m cliport.eval eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       exp_folder=exps 
```
`checkpoint_type`の指定方法は、`=test_best, =last, =05000`と指定する。

# デバッグの内容
## Dockerfile
・tqdmのスペルミス

・バージョン統合
## cliport\utils\utils.pyのImageRotatorクラス
オリジナルのutilsファイル(`cliport\utils\utils3.py`)は、Kornia（内部ではPyTorchのCUDA API）の`warp_affine`を使って画像をアフィン変換しようとした際に、CUDA内部エラー（cusolverエラー）が出た。
環境（PyTorchとCUDAのバージョン相性）によるバグで、CLIportのissueでもよく見る。

初めに、PyTorch標準のaffine_grid + grid_sampleに変えた(`cliport\utils\utils2.py`)を使用して、warp（画像変形）を行うことでcusolver errorは回避できた。しかし、ちょっと違う動きになって、モデル推論精度が落ちてしまった。

最終修正バージョンでは、Korniaを完全に排除し、アフィン行列を正しく組み立てた。ピボット中心を考慮してアフィン変換の並進も入れた。また、torch.pi も対応していないため、直接 πの数値 (3.1415...) を書く必要があった。これによって正常動作した。




