# データ収集
- 幅広いレベルかつプレースタイル動画を収集する

# データ獲得
## yt-dlpの使い方
### 基本コマンド
```bash
yt-dlp "動画のURL"
```
**品質指定**
```bash
# 最高画質で取得
yt-dlp -f "best" "URL"
# 720pで取得
yt-dlp -f "best[height<=720]" "URL"
# 1080pで取得
yt-dlp -f "best[height<=1080]" "URL"
```
**形式指定**
```bash
# MP4形式で保存
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" "URL"
```
**保存先とファイル名**
```bash
# 保存先ディレクトリ指定
yt-dlp -o "~/Downloads/%(title)s.%(ext)s" "URL"
# カスタムファイル名
yt-dlp -o "tabletennis_video.mp4" "URL"
```
### 卓球動画収集向けの推奨コマンド
```bash
# 1080p MP4形式で、タイトル付きで保存
yt-dlp -f "best[height<=1080][ext=mp4]" \
       -o "raw_02_training/%(title)s_%(id)s.%(ext)s" \
       "動画URL"
```