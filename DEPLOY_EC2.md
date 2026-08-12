# EC2 デプロイ手順書 — chap_md5_bulldozer

## 0. インスタンス選定

| インスタンス | GPU | 3080比 | 概算$/h(オンデマンド) | メモ |
| --- | --- | --- | --- | --- |
| **g5.xlarge** | A10G ×1 | 約1.0× | 約$1.0 | `make` が `sm_86` を自動検出。まずコレ |
| g6e.xlarge | L40S ×1 | 約2.5〜3× | 約$1.9 | 単一GPU最速。`make` が `sm_89` を自動検出 |
| g6.xlarge | L4 ×1 | 約0.7× | 約$0.8 | 安いが遅め |

> スポットなら6〜7割引。中断耐性のある用途なので **Spot 推奨**。
> **マルチGPU** が要るほど探索範囲が広いなら `g5.12xlarge`(A10G×4) / `g6e.12xlarge`(L40S×4) /
> `g6e.48xlarge`(L40S×8) など。本体は `--gpus N` / `--devices 0,2,3` で複数GPUを使える（手順7参照）。

## 1. インスタンスを起動（マネジメントコンソール）

**事前: ローカルで鍵を生成（初回のみ）** — EC2のキーペア機構は使わず、自分の公開鍵を
インスタンスの `~/.ssh/authorized_keys` に登録する方式にする。

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\ec2_alma
Get-Content $env:USERPROFILE\.ssh\ec2_alma.pub
```

1. EC2 → **Launch instance**
2. **AMI**: 「AMIカタログ」で `AlmaLinux OS 10`（発行元 AlmaLinux OS Foundation）を検索して選択
   （素の OS。ドライバ/CUDA は入っていないので手順3で導入する）
3. **Instance type**: `g5.xlarge`
4. **Key pair**: **「キーペアなしで続行 (Proceed without a key pair)」** を選択（pemは使わない。鍵は下の User data で登録）
5. **Network settings**: Security group で **SSH (TCP 22) を自分のIPのみ許可**
6. **Storage**: 既定の gp3 45〜100GB で十分
7. **Advanced details → User data** に貼り付け（cloud-init が起動時に `ec2-user` の `~/.ssh/authorized_keys` へ書き込む）:

```yaml
#cloud-config
ssh_authorized_keys:
  - ssh-ed25519 AAAA...（ec2_alma.pub の中身をそのまま1行）
```

   （Spot も同じ Advanced details 内で設定可）
8. **Launch instance** → 起動後、パブリックIPをメモ

## 2. SSH 接続

```powershell
ssh -i $env:USERPROFILE\.ssh\ec2_alma ec2-user@<PUBLIC_IP>
```

## 3. ドライバ + CUDA を導入（インスタンス内）

```bash
# 1) システム更新 → 最新カーネルで再起動
sudo dnf upgrade -y
sudo reboot
```

```bash
# 2) 再接続後: ビルドツールとカーネルヘッダ
sudo dnf install -y epel-release
sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r) gcc gcc-c++ make wget tmux dnf-plugins-core vim

# 3) NVIDIA CUDA リポジトリ (rhel10 = AlmaLinux 10)
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo
sudo dnf clean all

# 4) オープンGPUカーネルモジュール版ドライバ + CUDA Toolkit
sudo dnf -y install nvidia-open cuda-toolkit
sudo reboot
```

> パッケージ名が合わない場合は AWS re:Post の EC2×AlmaLinux 10 ガイド参照:
> https://repost.aws/articles/ARpmJcNiCtST2A3hrrM_4R4A/
> 代替として `sudo dnf module install nvidia-driver:open-dkms` でも可。

## 4. 動作確認と PATH

```bash
nvidia-smi          # A10G が見えればOK
# nvcc に PATH を通す
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version      # -arch=native は nvcc 11.5+ が必要（rhel10 リポジトリは CUDA 13.x）
```

## 5. ソース + Makefile を転送

**方法A: ローカルから scp**
```powershell
# ソースと Makefile の両方を転送
scp -i C:\path\to\mykey.pem `
  C:\Users\admin\voip\chap_md5_bulldozer.cu C:\Users\admin\voip\Makefile `
  ec2-user@<PUBLIC_IP>:~
```

**方法B: その場で貼り付け** — `vim chap_md5_bulldozer.cu` と `vim Makefile` で保存でも可。

## 6. ビルド

```bash
# 実行ファイル chap_md5_bulldozer を生成
make
# arch を固定したい場合（通常は不要）:
make ARCH=sm_86
```

> gcc が新しすぎて CUDA に "unsupported GNU version" と言われたら `make CCBIN=g++-13`（存在する g++ を指定）。

## 7. 実行

```bash
# 長い hex は変数に入れておくと楽
ID=123
CH=abcd1234ef567890
TG=abcd1234ef567890

# ① 検算（GPU不要）: --test は --target 不要、--id/--challenge は必須
./chap_md5_bulldozer --id $ID --challenge $CH --test PPPe0Tel@OLS

# ② 既定の探索（a-z0-9, 長さ1〜8）
./chap_md5_bulldozer --id $ID --challenge $CH --target $TG

# ③ 長さを広げる
./chap_md5_bulldozer --id $ID --challenge $CH --target $TG --min 1 --max 10

# ④ 文字集合を広げる（クラス指定: a-z + A-Z + 0-9 + 記号）
./chap_md5_bulldozer --id $ID --challenge $CH --target $TG --charset-classes aA0! --max 8
#   正規表現風でも可: --charset-regex "a-z0-9"

# ⑤ マルチGPUインスタンスの場合
./chap_md5_bulldozer --id $ID --challenge $CH --target $TG --gpus 4        # 先頭4基
./chap_md5_bulldozer --id $ID --challenge $CH --target $TG --devices 0,2,3 # 明示指定
```

> 既定空間(36^8)は A10G でも数分以内に全探索完了。長時間ジョブは `tmux` 内で
> 実行しておくと SSH 切断でも継続する（`tmux` は手順3で導入済み）。起動は `tmux`。

## 8. 後片付け ⚠️ 課金停止を忘れない

GPU インスタンスは高額。終わったら必ず停止/終了する。

- **一時中断（また使う）**: コンソールで **Stop**（EBS のみ少額課金）
- **完全に終わり**: **Terminate**（インスタンス削除、課金ゼロ）

CLI で終了する場合:
```powershell
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxx --region us-east-1
```

## CLI だけで一括起動（上級者向け）

```powershell
# 最新の AlmaLinux OS 10 の AMI ID を取得（発行元 AlmaLinux OS Foundation）
$AMI = aws ec2 describe-images --owners 764336703387 `
  --filters "Name=name,Values=AlmaLinux OS 10*x86_64*" "Name=architecture,Values=x86_64" `
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text --region us-east-1

# 事前に cloud-init.yaml を用意（手順1-7 と同じ #cloud-config、公開鍵を authorized_keys へ登録）
#   #cloud-config
#   ssh_authorized_keys:
#     - ssh-ed25519 AAAA...（ec2_alma.pub の中身）

# 起動（キーペアは使わず --user-data で公開鍵を注入。SG-ID は自分の値に）
aws ec2 run-instances --image-id $AMI --instance-type g5.xlarge `
  --security-group-ids sg-xxxxxxxx --region us-east-1 `
  --user-data file://cloud-init.yaml `
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=chap-crack}]'
# 起動後は手順3 のドライバ+CUDA導入が必要
```

## トラブルシュート

| 症状 | 対処 |
|---|---|
| `nvidia-smi` が失敗/No devices | ドライバ導入後の **reboot 未実施**、または `kernel-devel` が稼働カーネルと不一致(手順3) |
| `nvcc: command not found` | `export PATH=/usr/local/cuda/bin:$PATH`（手順4） |
| `provide --id / --challenge / --target` で終了 | これらは**必須引数**。手順7の変数(ID/CH/TG)を付けて再実行 |
| `unsupported GNU version` (gcc が新しすぎ) | `make CCBIN=g++-13` など、CUDA対応の g++ を指定 |
| `-arch=native` で失敗 | ビルドは GPU インスタンス上で行う（native はローカルGPUを見て判定）。または `make ARCH=sm_86` |
| コンパイルで `/utf-8` エラー | `make` を使えば自動。手動 nvcc なら Linux では `/utf-8` を付けない |
| SSH `Permission denied (publickey)` | User data の公開鍵が `ec2_alma.pub` の中身と一致しているか、ユーザー名 `ec2-user`、鍵の権限(手順2)、SG(22番)を確認 |
| 実行が遅い/0 Mh/s | 通常は native で最適。`make ARCH=` で誤指定していないか確認（A10G=`sm_86`, L40S/L4=`sm_89`） |
