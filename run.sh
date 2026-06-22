#!/usr/bin/env bash

# # 実行したいコマンドを書く
python train_after.py --load_from_cfg True --cfg_file config/study/art_clipart_after.yml
# python train_before.py --load_from_cfg True --cfg_file config/study/art_clipart_before.yml

python train_after.py --load_from_cfg True --cfg_file config/study/clipart_realworld_after.yml
python train_before.py --load_from_cfg True --cfg_file config/study/clipart_realworld_before.yml

# python train_after.py --load_from_cfg True --cfg_file config/study/dslr_amazon_after.yml
# python train_before.py --load_from_cfg True --cfg_file config/study/dslr_amazon_before.yml