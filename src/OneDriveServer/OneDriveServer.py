import rawpy
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import asyncio
import traceback

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

intents = discord.Intents.all()
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
        await asyncio.sleep(2)
        
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

            img_corrected = np.clip(rgb_raw - BLACK_LEVEL, 0, None)
            img_bgr = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR)
            # 保存フォルダ作成
            os.makedirs(OneDrive_RAW_PNG_PATH, exist_ok=True)
            os.makedirs(OneDrive_GAMMA_PNG_PATH, exist_ok=True)
            
            # RAW PNG 保存
            filename_raw = os.path.splitext(os.path.basename(dng_path))[0] + ".png"
            save_path_raw = os.path.join(OneDrive_RAW_PNG_PATH, filename_raw)
            cv2.imwrite(save_path_raw, img_bgr)
            
            # 8bit + ガンマ補正
            rgb_gamma = to_8bit_gamma(img_bgr)
            filename_gamma = os.path.splitext(os.path.basename(dng_path))[0] + "_gamma.jpg"
            save_path_gamma = os.path.join(OneDrive_GAMMA_PNG_PATH, filename_gamma)
            cv2.imwrite(save_path_gamma, rgb_gamma, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            print(f"処理完了: {dng_path}")

    except Exception:
        tb = traceback.format_exc()
        print(f"エラー: {tb}")
        await send_error_to_discord(f"{dng_path} 処理中に例外発生:\n{tb}")

async def watch_folder(folder_path, processed_files):
    while True:
        try:
            files = set(f for f in os.listdir(folder_path) if f.lower().endswith(".dng"))
            new_files = files - processed_files
            for file in new_files:
                file_path = os.path.join(folder_path, file)
                asyncio.create_task(process_dng_async(file_path))
            processed_files.update(new_files)
            await asyncio.sleep(1)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"監視エラー: {tb}")
            await send_error_to_discord(f"フォルダ監視中に例外:\n{tb}")
            await asyncio.sleep(5)

@bot.event
async def on_ready():
    print(f"Bot 起動: {bot.user}")
    # フォルダ監視タスク開始
    processed_files = set()
    bot.loop.create_task(watch_folder(OneDrive_DATA_PATH, processed_files))


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)