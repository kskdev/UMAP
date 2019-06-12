# coding:utf-8

import argparse
from functools import reduce
from glob import glob
import os
import random

import numpy as np
from PIL import Image
import umap  # conda install -c conda-forge umap-learn (Win, Mac, Linux)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D


# データをスタックし，結合した後，次元圧縮
class Compress:
    def __init__(self, seed=-1):
        self.data = []
        self.target_data = None
        self.target_label = None
        self.compress_data = None
        self.out_data = None
        self.seed = seed

    # 1クラス分のデータを入力する (何回も呼び出し可能)
    # data: numpy array, cls_id: (int or float)
    def stack_data(self, data, cls_id):
        d = np.hstack((np.ravel(data), cls_id))
        self.data.append(d.tolist())

    # データを整列した後，次元圧縮
    def fitting(self, n_components=2, n_neighbors=15, min_dist=0.1):
        # 2次元リストからnumpy配列へ変換
        self.data = np.asarray(self.data, dtype=np.float32)
        # 入力データとラベルに分割
        self.target_data = self.data[:, 0:-1]
        self.target_label = np.reshape(self.data[:, -1], newshape=(-1, 1))

        # 次元圧縮の実行
        self.compress_data = umap.UMAP(n_neighbors, n_components, min_dist=min_dist).fit_transform(self.target_data)

        # 圧縮データにラベルを付与
        self.out_data = np.hstack((self.compress_data, self.target_label))
        return self.out_data


# 2次元上に点または画像を出力する(クラスラベルがあるとそれに対応する)
class Visualize:
    def __init__(self, fig_size=(10, 6), dpi=512, color_num=7, color_map='cm.Set1'):
        self.fig_size = fig_size
        self.dpi = dpi

        # カラーマップ / 画像の枠の太さの設定
        self.color_map = eval(color_map)  # Select Qualitative colormaps
        self.color_num = color_num
        self.cls_list = [i for i in range(self.color_num)]
        self.rgb_list = [self.color_map(float(i) / float(self.color_num)) for i in self.cls_list]
        self.frame_width = 5
        self.resize = None

    # 画像をグラフに出力する
    def draw_image2d(self, save_name, x_vec, y_vec, paths, labels, resize=(128, 128), zoom=1, frame_width=1):
        plt.figure()
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        self.resize = resize
        for _x, _y, _img, _lbl in zip(x_vec, y_vec, paths, labels):
            image = self.create_image(_img, _lbl, max(np.unique(labels)), frame_width)
            im = OffsetImage(image, zoom=zoom)
            ab = AnnotationBbox(im, (_x, _y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.plot(x_vec, y_vec, 'ko', alpha=0)
        plt.axis('off')
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    # 画像に image_label に対応した色の枠を付ける
    def create_image(self, img_path, img_label, class_num, frame_width):
        # class_numからフレームの色を決定する(カラーマップからRGB値を取得)
        r, g, b, a = self.rgb_list[img_label]
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        pil_img = Image.open(img_path).convert('RGB').resize(self.resize)
        img_w, img_h = int(pil_img.size[0]), int(pil_img.size[1])

        # 背景画像を設定(画像より一回り大きい背景画像の上に画像を載せることでフレームっぽくする)
        bg_w = img_w + (frame_width * 2)
        bg_h = img_h + (frame_width * 2)
        bg = Image.new('RGB', (bg_w, bg_h), (r, g, b))

        # img を bg に貼り付け(left, topはbgに貼り付ける時の左上座標)
        left = (bg_w - img_w) // 2
        top = (bg_h - img_h) // 2
        bg.paste(pil_img, (left, top))
        return np.asarray(bg, dtype=np.uint8)

    def draw_point2d(self, save_name, x_vec, y_vec, labels, label_name_list):
        # Ref : https://stackoverflow.com/questions/42056713/matplotlib-scatterplot-with-legend
        plt.figure()
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        cls_list = np.unique(labels)
        for i, u in enumerate(cls_list):
            xi = [x_vec[j] for j in range(len(x_vec)) if labels[j] == u]
            yi = [y_vec[j] for j in range(len(y_vec)) if labels[j] == u]
            ax.scatter(xi, yi, c=self.rgb_list[i], label=label_name_list[u])
        ax.legend()
        ax.axis('off')
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    def draw_point3d(self, save_name, x_vec, y_vec, z_vec, labels, label_name_list):
        # label_name_list format: 'Name A', 'Name B', 'Name C', ...}
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        ax = Axes3D(fig)

        cls_list = np.unique(labels)
        for i, u in enumerate(cls_list):
            xi = [x_vec[j] for j in range(len(x_vec)) if labels[j] == u]
            yi = [y_vec[j] for j in range(len(y_vec)) if labels[j] == u]
            zi = [z_vec[j] for j in range(len(z_vec)) if labels[j] == u]
            ax.scatter(xi, yi, zi, c=self.rgb_list[i], label=label_name_list[u])
        ax.legend()
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)


def func_read_img(file_path):
    return Image.open(file_path).convert('RGB').resize(read_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Mapping by (UMAP or TSNE)')
    # 最低限必要なもの
    parser.add_argument('--glob_path_list', type=str)
    parser.add_argument('--label_list', type=str)
    # オプション
    parser.add_argument('--save_dir', type=str, default='Result')
    parser.add_argument('--sampling_num', type=int, default=50)
    parser.add_argument('--is_random', type=str, default=False)
    parser.add_argument('--read_width', type=int, default=640 // 3)
    parser.add_argument('--read_height', type=int, default=480 // 3)
    parser.add_argument('--method_param_neighbor', type=int, default=10)
    parser.add_argument('--umap_param_min_dist', type=int, default=0.1)
    parser.add_argument('--umap_init_seed', type=int, default=-1)
    # Scatterの代わりに画像を貼り付ける場合のオプション
    parser.add_argument('--add_paste_image', type=bool, default=True)
    parser.add_argument('--frame_wid_pix', type=int, default=8)
    parser.add_argument('--paste_scale', type=int, default=0.1)
    # パーサを作成
    args = parser.parse_args()

    # 初期化
    save_dir = args.save_dir
    sample_num = args.sampling_num
    is_random = bool(args.is_random)
    read_size = (args.read_width, args.read_height)
    neighbor, min_dis = args.method_param_neighbor, args.umap_param_min_dist
    zoom = args.paste_scale
    frame_wid = args.frame_wid_pix
    os.makedirs(save_dir, exist_ok=True)

    # 読み込み用画像パスの取得
    glob_list = args.glob_path_list.split(',')
    print(glob_list)
    a = list(map(lambda g: glob(g)[:sample_num], glob_list))
    b = list(map(lambda g: random.sample(glob(g), sample_num), glob_list))
    path_lists = b if is_random else a
    # ラベルのリスト化
    label_list = args.label_list.split(',')

    # 画像の読み込んで，ラベルIDとともにcompress インスタンスにスタック
    compress = Compress(seed=args.umap_init_seed)
    for i, p_list in enumerate(path_lists):
        for j, f in enumerate(p_list):
            compress.stack_data(func_read_img(f), i)
            print('Class {:>3}: Read {:>6}/{:<6}'.format(i, j + 1, len(p_list)), '\r', end='')

    print('\n')
    # 次元圧縮の実行(2次元)
    print('Compressing data (2-Dimension)...')
    out = compress.fitting(n_components=2, n_neighbors=neighbor, min_dist=min_dis)
    print('Done!')
    x, y, t = np.split(out, 3, axis=1)
    x, y, t = np.ravel(x), np.ravel(y), np.ravel(t.astype(np.int))
    save_path = os.path.join(save_dir, '2dimMap')
    print('Drawing 2dim Mapping...')
    Visualize().draw_point2d(save_path + '.png', x, y, t, label_list)
    Visualize().draw_point2d(save_path + '.pdf', x, y, t, label_list)
    print('Done!')

    if args.add_paste_image:
        all_path = reduce(lambda pa, pb: pa + pb, path_lists)
        save_path = os.path.join(save_dir, '2dimImage')
        print('Drawing 2dim Image Mapping...')
        Visualize().draw_image2d(save_path + '.png', x, y, all_path, t, read_size, zoom, frame_wid)
        print('Done!')

    # 次元圧縮の実行(3次元)
    print('Compressing data (3-Dimension)...')
    out = compress.fitting(n_components=3, n_neighbors=neighbor, min_dist=min_dis)
    print('Done!')
    x, y, z, t = np.split(out, 4, axis=1)
    x, y, z, t = np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t.astype(np.int))
    save_path = os.path.join(save_dir, '3dimMap')
    print('Drawing 3dim Mapping...')
    Visualize().draw_point3d(save_path + '.png', x, y, z, t, label_list)
    Visualize().draw_point3d(save_path + '.pdf', x, y, z, t, label_list)
    print('Done!')
