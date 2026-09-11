# Kaggle Draft 操作與問題總結

更新日期：2026-09-12。最後一次操作已取消 TPU 排隊，Active Events 顯示 0；本機監控已停止。本次整理不啟動 TPU。

## 使用哪份 Notebook

匯入 `notebook/qwen38-tpu-draft.ipynb` 至自己的 Kaggle Draft。這份檔案由 `tools/sync_notebooks.py` 產生，包含設定、完整 serving script 與啟動 cell，沒有歷史輸出或臨時字串替換修補 cell。

既有 Kaggle 網頁 Draft 曾出現儲存失敗，後來重新開啟時仍是舊 HTTP/2 patch。Git push 不會更新 Kaggle 編輯器；下次啟動前必須匯入新版 Notebook，保留／重新確認 Inputs 與 Secrets。不要把舊 patch cell 複製回來。

## 啟動

1. 先在 Chrome 前景開啟 Kaggle Draft 編輯頁，確認帳號與 Notebook。
2. 設 Accelerator 為 TPU VM v5e-8，Internet 為 ON。
3. 掛載模型 `king88888888/qwen38-27b-uncensored-bf16` 與環境快取 `rahim3/qwen38-tpu-env-v5e8`。不需要原作者的原版 Qwen weights Dataset。
4. 在 Kaggle Secrets 授權 `KTL_API_KEY`、`CF_TUNNEL_TOKEN`。不要把實際金鑰貼入 Notebook、Git 或日誌文檔。
5. 按 Run All，使用 Interactive Draft；Save Version／CLI push 是另外一種執行流程。
6. 等待排隊、掛載、runtime、cache、weights、vLLM、tunnel、warm-up。先讓自測完成，再連 Hermes，避免同時請求影響測速。

預設模型是 bf16 `JonathanColetti/Qwen3.8-27B-Uncensored`，context 262144、4 concurrent sequences、MTP 0、fast_start true、reasoning xhigh、keepalive 180 分鐘。固定 endpoint 為 `https://qwen.aceinifnity.com/v1`，模型名稱為 `qwen3.8-27b-uncensored`。Cloudflare published application 指向 `http://localhost:8000`。

模型 Dataset 已掛載時，STEP 3 應讀取本地檔案；`weights-download` 表示沒有找到可用的掛載模型。Dataset 掛載與複製權重到 TPU 仍需要時間，不能把它們都解讀為 Hugging Face 重新下載。額度以 Kaggle 帳號頁面為準，不能單憑 Starting 判斷是否計費／扣額度。

## 排隊監控

macOS Chrome 必須保持開啟，並允許 Apple Events 執行 JavaScript。在本機 `~/.config/kaggle-tpu-lab/queue-monitor.json` 放以下設定，替換自己的通知 key，檔案不要加入 Git：

```json
{
  "tab_url_contains": "kaggle.com/code/king88888888/qwen3-8-27b-uncensored-interactive-tpu/edit",
  "notify_url_template": "https://api.day.app/YOUR_KEY/Kaggle/{content}"
}
```

```bash
python tools/monitor_kaggle_queue.py --interval 180 --threshold 5
```

第 5 位以內通知一次後退出。數字是順位，前方人數為順位減一。沒有數字時保持監控，不推算順位。曾發生編輯器顯示 Starting、Active Events 卻仍顯示 Queued；舊腳本因此誤報「已排完」。新版不把 Starting 當成排完隊，只有明確 Running 才做啟動通知。電腦睡眠或 Chrome 關閉會影響監控。Ctrl-C 只停止監控。

## Cloudflare 與速度問題

舊環境 binary 曾以 SIGSEGV（exit -11）退出；換成官方下載的 cloudflared 2026.9.0 後曾成功連通。較早日誌亦出現 invalid token，不能把所有退出都歸咎於同一原因。新版每次同步下載、執行 `--version` 驗證後才原子替換；失敗就停止啟動，不回退到 Dataset 裡的舊 binary。

協議預設 auto；連接程序退出時記錄 exit code 與尾端日誌，重試在 auto／http2 間切換。HTTP/2 是 TCP 傳輸選項，不能修復無效 token 或崩潰 binary。wrapper 的重連訊息不足以定位原因，必須看 `/kaggle/working/vllm.log` 的 cloudflared 原始錯誤。

本次歷史記錄曾測到約 74.7 tok/s，也曾只有 3.6 tok/s；低速原因尚未確認。內建 benchmark 呼叫 localhost，排除首 token 等待後計算 192-token 輸出的 decode 速率，因此 Cloudflare 傳輸不能直接解釋內建測速下降。下一次需在沒有 Hermes 請求時重測，檢查 vLLM 日誌、實際 runtime／TPU 與編譯情況。這次本機整理沒有重新跑 TPU，不能宣稱效能已恢復。

READY 只代表本地 vLLM 健康，不保證外部 tunnel 已可用。外部 `/models` 未帶金鑰返回 401 僅證明到達驗證層；仍需帶金鑰測 chat 與 streaming 才能驗證完整服務。

## 停止與維護

用 Active Events → Stop Session，確認 Cancelled 和 0 Active Events，再 Ctrl-C 停止監控。關閉 Chrome 分頁不等於停止遠端 Session。`keepalive_min` 目前從 warm-up／自測後起算，只結束服務程式，不保證 Kaggle 釋放整個 VM；睡前仍要手動確認停止。

修改 `kernel/serve_qwen38.py` 後執行 `python tools/sync_notebooks.py`，同步所有 Notebook 並清除輸出。Git 倉庫是來源；網頁 Draft 需另外匯入。下次啟動順序：匯入、確認設定與 Secrets、前景 Run All、啟動監控、等待自測完成、測試 Hermes。
