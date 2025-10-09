import rawpy
import numpy as np
import cv2
import os
import asyncio
import traceback
import discord
from discord.ext import commands
from dotenv import load_dotenv
import signal

# OneDriveフォルダー設定
load_dotenv()
OneDrive_DATA_PATH = os.getenv("OneDrive_DATA_PATH")
OneDrive_RAW_PNG_PATH = os.getenv("OneDrive_RAW_PNG_PATH")
OneDrive_GAMMA_PNG_PATH = os.getenv("OneDrive_GAMMA_PNG_PATH")
Onedrive_TRASH_BOX_PATH = os.getenv("Onedrive_TRASH_BOX_PATH")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# カメラ設定（black/whiteレベル）
BLACK_LEVEL = 528
WHITE_LEVEL = 4095

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

async def send_error_to_discord(error_message: str):
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        await channel.send(f"❌ エラー発生:\n```\n{error_message}\n```")

def to_8bit_gamma(img, gamma=2.2):
    """
    12bitまたは16bit画像を8bitに変換して、ガンマ補正も適用（表示用）
    """
    # 正規化（0〜1）
    img = np.clip((img)/ (WHITE_LEVEL - BLACK_LEVEL), 0, 1)

    # ガンマ補正（sRGB風）
    img_gamma = np.power(img, 1 / gamma)

    # 8bit化
    return (img_gamma * 255).astype(np.uint8)

async def process_dng_async(dng_path):
    try:
        # ファイルが完全に書き込まれるのを待つ
        await asyncio.sleep(1)
        # ブラックラベル補正も自動でやっている
        with rawpy.imread(dng_path) as raw:
            rgb_raw = raw.postprocess(
                use_camera_wb=False,
                no_auto_bright=True,
                no_auto_scale=True,
                output_bps=16,
                gamma=(1,1),
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                output_color=rawpy.ColorSpace.raw,
                half_size=True
            )

            # 保存フォルダ作成
            os.makedirs(OneDrive_RAW_PNG_PATH, exist_ok=True)
            os.makedirs(OneDrive_GAMMA_PNG_PATH, exist_ok=True)
            
            # RAW PNG 保存
            filename_raw = os.path.splitext(os.path.basename(dng_path))[0] + ".png"
            save_path_raw = os.path.join(OneDrive_RAW_PNG_PATH, filename_raw)
            cv2.imwrite(save_path_raw, cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR))
            
            # 8bit + ガンマ補正
            rgb_gamma = to_8bit_gamma(rgb_raw)
            filename_gamma = os.path.splitext(os.path.basename(dng_path))[0] + "_gamma.jpg"
            save_path_gamma = os.path.join(OneDrive_GAMMA_PNG_PATH, filename_gamma)
            cv2.imwrite(save_path_gamma, cv2.cvtColor(rgb_gamma, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            print(f"処理完了: {dng_path}")

    except Exception:
        tb = traceback.format_exc()
        print(f"エラー: {tb}")
        await send_error_to_discord(f"{dng_path} 処理中に例外発生:\n{tb}")

async def watch_folder(folder_path):
    processed_files = set(os.listdir(folder_path)) 
    print(f"監視開始: {folder_path}（既存 {len(processed_files)} 件をスキップ）")
    while True:
        try:
            current_files = set(f for f in os.listdir(folder_path) if f.lower().endswith(".dng"))
            new_files = current_files - processed_files

            for file in new_files:
                file_path = os.path.join(folder_path, file)
                print(f"新規ファイル検出: {file_path}")
                asyncio.create_task(process_dng_async(file_path))

            processed_files.update(new_files)
            await asyncio.sleep(1)

        except Exception:
            tb = traceback.format_exc()
            print(f"監視エラー: {tb}")
            await send_error_to_discord(f"フォルダ監視中に例外:\n{tb}")
            await asyncio.sleep(5)


# ========= Discord 通知 =========
async def send_to_discord(message: str):
    """共通のDiscord送信関数"""
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        await channel.send(message)
    else:
        print("⚠️ Discordチャンネルが見つかりません。IDを確認してください。")

# ========= Botイベント =========
@bot.event
async def on_ready():
    print(f"Bot 起動: {bot.user}")
    await asyncio.sleep(1)
    await send_to_discord(f"🚀 Bot 起動完了: **{bot.user}** がオンラインになりました！")

    asyncio.create_task(watch_folder(OneDrive_DATA_PATH))


# ========= 終了処理 =========
async def shutdown():
    print("🛑 終了処理中...")
    await send_to_discord("🛑 Botがシャットダウンされます。処理を停止します。")
    await bot.close()
    print("✅ Bot終了完了")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)