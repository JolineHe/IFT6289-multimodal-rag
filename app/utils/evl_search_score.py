#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Miao at 2025-03-30
'''
@File   : evl_search_score.py
@Time   : 2025-03-30 5:35 p.m.
@Author : Miao
@Project: IFT6289-multimodal-rag 
@Desc   : Please enter here
'''
import os.path

import matplotlib.pyplot as plt
import requests
from io import BytesIO
import pandas as pd
from PIL import Image


def check_pic_result(df, savepath='../data', save_title=''):
    # import requests

    # Determine Grid Size (Auto-adjust for number of images)
    num_images = len(df)
    cols = min(3, num_images)  # Max 3 columns
    rows = (num_images // cols) + (num_images % cols > 0)

    # Create Subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten axes if more than one row
    axes = axes.flatten() if num_images > 1 else [axes]

    # Display Each Image with Score as Caption
    for i, row in df.iterrows():
        axes[i].imshow(row["image"])
        axes[i].axis("off")  # Hide axes
        axes[i].set_title(f"Score: {row['score']:.2f}", fontsize=12)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    plt.savefig(f'{savepath}/{save_title}.png', dpi=60)

    # plt.show()


def is_remote_image_exist(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)  # Use HEAD for efficiency
        return response.status_code == 200  # True if the image exists
    except requests.RequestException:
        return False  # Handle network issues or invalid URLs


def load_image(url):
    if url.startswith("http"):  # Remote image
        if not is_remote_image_exist(url):
            return Image.new("RGB", (512, 512), (0, 0, 0))
        else:
            response = requests.get(url)
            return Image.open(BytesIO(response.content))
    else:  # Local image
        return Image.open(url)


def evl_search_result(results, img_gt_path, title='text',savepath='../data'):
    print('===============================')
    results_pics = ['' if i.get('images').get('picture_url') is None else i['images']['picture_url'] for i in results]
    score_name = 'search_score'
    if title.find('by_text')>=0:
        score_name = 'text_search_score'
    elif title.find('by_image')>=0:
        score_name = 'image_search_score'
    scores = [float(i[score_name]) for i in results]
    df_out = pd.DataFrame({
        'score': scores + [1],
        'image_url': results_pics + [img_gt_path]}
    )
    # Load images
    df_out["image"] = df_out["image_url"].apply(load_image)
    df_out = df_out.sort_values(by="score", ascending=True)

    check_pic_result(df_out, save_title=title, savepath = savepath)