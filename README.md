# bonsai-27b-wsl-tp2

**Ternary-Bonsai-2-27B**(27B、重み2bit)を、**12GB の GPU 2枚**(RTX 5070 × 2、WSL2)で、**素の vLLM 0.34.0** とプラグインで動かすための一式です(vLLM 本体は書き換えません)。

- OpenAI 互換 API(思考・ツール呼び出し・画像入力)
- 文脈 131,072 トークン、同時 24 本
- 単独 約95 tok/s、24本同時で合計 約470 tok/s(1本あたり約21 tok/s、手元の実測・vLLM 0.34.0)

解説記事: [12GBのGPU2枚で27BのLLMを24人同時に使う(Qiita)](https://qiita.com/nakanokensetsu/items/66e136acefd5a54b2f1d)

## 中身

| パス | 内容 |
|---|---|
| `setup.sh` | venv 作成 → vLLM 0.34.0 → `bonsai_ext` → プラグインのフォークを入れる |
| `requirements-lock.txt` | 動作確認した環境のパッケージの版(vLLM 0.34.0、torch 2.14.0+cu130 など約200個)をすべて固定 |
| `bonsai_ext/` | vLLM のプラグイン(`vllm.general_plugins`)。OSCAR KV(回転+低ビット)、優先度による追い出し(`--scheduler-cls`)、prefill のステップ上限、埋め込みの SSD 化、ビジョンタワーの RAM 退避、WSL2 での KV 退避の修正、TurboQuant 高速化([vllm#53060](https://github.com/vllm-project/vllm/pull/53060)・[#53410](https://github.com/vllm-project/vllm/pull/53410) の移植)、VRAM の記録(`--worker-cls`) |
| `model/prepare_model.sh` | 重みのダウンロード+画像用のビジョンタワー追加+[日本語で学習し直した MTP ヘッド](https://huggingface.co/nakakennnn/Ternary-Bonsai-2-27B-MTP-ja)への差し替え+埋め込みの書き出し |
| `serve.sh` | 起動スクリプト。機能はすべて環境変数でオン/オフ |
| `script_ban/` | 指定した文字種(キリル文字など)を出さない logits processor |
| `ssd_embed/` | 埋め込みを SSD に置くための書き出しスクリプトと、GPU に残す語の一覧 |
| `turboquant_ext/` | (任意)`turboquant_4bit_nc` を使う場合の CUDA 高速経路(`bonsai_ext` に入る) |

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
| `GDN_PREFILL` | `triton` | GDN 層の prefill カーネル。`flashinfer` は要求ごとに途中保存の枠を取るため、手元では KV プールが約9%減り、速度は同じでした |
| `TEMPLATE` | なし | チャットテンプレート。私は [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) を使っています |

要求ごとの優先度は、リクエストの JSON に `"priority": 0`(小さいほど先)を入れます。文字種の禁止は `"vllm_xargs": {"ban_scripts": "cyr,arab,thai,hangul,han_simp"}` です。

## 注意

- 非公式です。Prism ML・vLLM プロジェクトとは関係ありません。
- `bonsai_ext` は vLLM **0.34.0 で確認**しています。一部は vLLM 内部のモジュールを差し替えるため、ほかの版では動かない可能性があります。
- vLLM 0.32.0 にパッチを当てる旧版は [`v0.32` タグ](https://github.com/Nakanokensetsu/bonsai-27b-wsl-tp2/tree/v0.32)にあります。
- 出力が素の vLLM と完全に一致するかは確認していません。量子化(KV・int8)による差があります。

## ライセンス

Apache-2.0。vLLM(Apache-2.0)由来のファイル(`bonsai_ext/src/bonsai_ext/vendor/`)と、vllm#53060・#53410 由来のコードを含みます。モデルの重みは各配布元のライセンスに従ってください。
