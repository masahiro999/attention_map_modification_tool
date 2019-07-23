# Attention map modification tool 
Writer : [Masahiro Mitsuhara](https://github.com/Masahiro-Mitsuhara)<br>
モデルを評価して生成したAttention mapを任意に修正することができます．

## 動作環境
Ubuntu 16.04 LTSにて動作確認済みです．<br>
Dockerで環境を構築する場合はDockerfileを使用して環境を構築します.
- PyTorch : 0.4.0
- PyTorch vision : 0.2.1
- Flask : 1.0.3
- opencv-python : 3.4.1.15

以下のコマンドで必要なパッケージをインストールしてください．<br>
torch のインストールは，[こちらのリンク](https://pytorch.org/get-started/locally/)に従ってください．

```sh
pip3 install opencv-python
pip3 install flask
pip3 install torch
pip3 install torchvision
```

## 実行方法
GPUを使用して，学習済みモデルからAttention mapの生成を行います．<br>
次に，Attention mapの修正ツールが起動する．<br>
実行方法がGPUが使用可能かどうかで変わります．<br>
GPUが使用可能な場合は学習済みのモデルを評価し，任意の画像のAttention mapを生成することができます．<br>
GPUを使用する環境がない場合は，修正ツールのみを実行してください．<br>
サンプルの画像が用意してあるので、修正ツールをお試し頂くことができます．<br>
GPUを使用して，学習済みモデルからAttention mapの生成を行います．<br>
次に，Attention mapの修正ツールが起動する．<br>
下記のコマンドを実行してください．<br>
※モデルを評価する環境がない場合は，修正ツールのみを実行する．
- Attention mapの生成と修正ツールの起動 
> python3 main.py -a resnet152 --data data/ -c checkpoints/results --gpu-id 0 --evaluate --resume checkpoints/results/model_best.pth.tar

- 修正ツールの起動
> python3 edit_attention.py

### モデルのダウンロード
モデルを下記のリンクからダウンロードしてください．
ダウンロードしたモデルをcheckpoints/results/以下に置いてください．
- ResNet152 : https://drive.google.com/open?id=1kMfRApXYsHEa53SjxXzCMs3lNaC8T2aH

### Attention mapの修正方法 
Attention mapの修正は，マウス操作とキーボード操作で行います.<br>
ツールの起動時に表示されるパレットでペンの太さと色を調節できます.
- Attentionの追加 : Altキーを押しながら、マウスを動かす
- attentionの削除 : Shiftキーを押しながら、マウスを動かす
- ツールの終了 : Escキー
- Attention mapの保存 : sキー
- 次の画像に移動 : nキー
- 一つ前の画像に戻る : bキー
- 編集のやり直し : rキー
- 一つ前の編集過程に戻る : qキー
- ウェブ上で編集結果を確認 : cキー

### Attention mapの修正例
![overview image](https://github.com/Masahiro-Mitsuhara/attention_map_modification_tool/blob/master/example.jpg)

