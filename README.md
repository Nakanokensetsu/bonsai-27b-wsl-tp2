# bonsai-27b-wsl-tp2

**Ternary-Bonsai-2-27B**(27B、重み2bit)を、**12GB の GPU 2枚**(RTX 5070 × 2、WSL2)で vLLM 0.32.0 から動かすための一式です。

- OpenAI 互換 API(思考・ツール呼び出し・画像入力)
- 文脈 131,072 トークン、同時 24 本
- 単独 約90 tok/s、24本同時で合計 約600 tok/s(手元の実測)

解説記事: (Qiita の URL)

## 中身

| パス | 内容 |
|---|---|
| `setup.sh` | venv 作成 → vLLM 0.32.0 → パッチ → プラグインのフォークを入れる |
| `requirements-lock.txt` | 動作確認した環境のパッケージの版(vLLM 0.32.0、torch 2.13.0 など200個)をすべて固定 |
| `vllm-patch/vllm-0.32.0-bonsai.patch` | vLLM 0.32.0 への差分(27ファイル)。OSCAR KV(回転+低ビット)、優先度による追い出し、prefill のステップ上限、埋め込みの SSD 化、WSL2 での KV 退避の修正、TurboQuant 高速化([vllm#53060](https://github.com/vllm-project/vllm/pull/53060) の移植)など |
| `model/prepare_model.sh` | 重みのダウンロード+画像用のビジョンタワー追加+埋め込みの書き出し |
| `serve.sh` | 起動スクリプト。機能はすべて環境変数でオン/オフ |
| `script_ban/` | 指定した文字種(キリル文字など)を出さない logits processor |
| `ssd_embed/` | 埋め込みを SSD に置くための書き出しスクリプトと、GPU に残す語の一覧 |
| `turboquant_ext/` | (任意)`turboquant_4bit_nc` を使う場合の CUDA 高速経路 |

プラグイン: [Nakanokensetsu/bonsai-vllm](https://github.com/Nakanokensetsu/bonsai-vllm/tree/bonsai-tp2-wsl)(`bonsai-tp2-wsl` ブランチ)。[fraserprice/bonsai-vllm](https://github.com/fraserprice/bonsai-vllm) のフォークで、TP=2 対応・int8 prefill・dp4a decode・int8 AllReduce を足したものです。

## 動作確認した環境

| 項目 | 内容 |
|---|---|
| GPU | RTX 5070 12GB × 2(sm_120) |
| OS | Windows 11 + WSL2(Ubuntu)、RAM 64GB を WSL に割り当て |
| ドライバ | CUDA 13.0 対応の NVIDIA ドライバ(Windows 側) |
| Python | 3.12(uv が用意する) |

ほかの GPU では未確認です。sm_120 以外では `TORCH_CUDA_ARCH_LIST` を変えてください。

## 手順

```bash
git clone https://github.com/Nakanokensetsu/bonsai-27b-wsl-tp2.git
cd bonsai-27b-wsl-tp2
./setup.sh                  # 15〜30分(ダウンロード約8GB)
./model/prepare_model.sh    # 重み 約10.5GB
./serve.sh                  # 初回は 3〜5 分(カーネルのコンパイルと CUDA グラフ)
```

`Application startup complete.` が出たら、別のターミナルで:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-27b","messages":[{"role":"user","content":"17×23は?"}]}' | python3 -m json.tool
```

## 機能のオン/オフ(serve.sh の環境変数)

| 変数 | 既定 | 内容 |
|---|---|---|
| `KV` | `turboquant_k3v2_oscar` | KV の形式。`turboquant_k2v2_oscar`(さらに小さい、数学で精度低下)/`turboquant_4bit_nc`/`auto` |
| `MEM` | 0.72 | GPU メモリの割合。画面表示と兼用の GPU では下げる |
| `SEQS` | 24 | 同時に処理する本数 |
| `PRISM_AR_INT8` | 1 | prefill の AllReduce を int8 で |
| `VLLM_PRIO_PREEMPT` | 1 | `priority` の小さい要求が待っていたら、実行中の大きい要求を追い出す |
| `VLLM_SSD_EMBED_DIR` | `./embed_ssd` | 空にすると埋め込みを全部 GPU に置く |
| `KV_OFFLOAD` | 1 | GPU から追い出された KV を RAM 2GiB → SSD に退避 |
| `TEMPLATE` | なし | チャットテンプレート。私は [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) を使っています |

要求ごとの優先度は、リクエストの JSON に `"priority": 0`(小さいほど先)を入れます。文字種の禁止は `"vllm_xargs": {"ban_scripts": "cyr,arab,thai,hangul,han_simp"}` です。

## 注意

- 非公式です。Prism ML・vLLM プロジェクトとは関係ありません。
- パッチは vLLM **0.32.0 専用**です。ほかの版には当たりません。
- 出力が素の vLLM と完全に一致するかは確認していません。量子化(KV・int8)による差があります。

## ライセンス

Apache-2.0。vLLM(Apache-2.0)への差分と、vllm#53060 由来のコードを含みます。モデルの重みは各配布元のライセンスに従ってください。
