#!/usr/bin/env bash
set -e

GLOB_LIST_STR=\
'/data1/Dataset/cityScapes/leftImg8bit/train/*/*leftImg8bit.png',\
'/data1/Dataset/GTA/Dataset/*/images/*.png'
LABEL_LIST_STR='Cityscapes','GTAV'

SAVE_DIR='city_gtav'
SAMPLE_NUM=200
IS_RANDOM=True
READ_WIDTH=320
READ_HEIGHT=240
UMAP_SEED=-1
mkdir -p ${SAVE_DIR}
cat $0 > ${SAVE_DIR}/args.out

python compress.py \
--glob_path_list ${GLOB_LIST_STR} \
--label_list ${LABEL_LIST_STR} \
--save_dir ${SAVE_DIR} \
--sampling_num ${SAMPLE_NUM} \
--is_random ${IS_RANDOM} \
--read_width ${READ_WIDTH} \
--read_height ${READ_HEIGHT} \
--umap_init_seed ${UMAP_SEED}
