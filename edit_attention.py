# -*- coding: utf-8 -*-
import cv2
import os
import glob
import numpy as np
import subprocess
from time import sleep
from tqdm import tqdm

def min_max(x, axis=None):
    x = np.array(x, dtype=np.float32)
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    result = result * 255.
    return result

def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)

    zscore = (x-xmean)/xstd
    return zscore

# sx, syは線の始まりの位置
sx, sy, sx2, sy2, sx3, sy3 = 0, 0, 0, 0, 0, 0

# ペンの色
color = (0, 0, 0)

# ペンの太さ
thickness = 1

# マウスの操作があるとき呼ばれる関数
def callback(event, x, y, flags, param):
    global sx, sy, sx2, sy2, sx3, sy3, color, thickness, att_gray

    # マウスの左ボタンがクリックされたとき
    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy = x, y

    # マウスの左ボタンがクリックされていて、マウスが動いたとき
    if flags ==  cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE:
        cv2.line(att_gray, (sx, sy), (x, y), color, thickness)
        sx, sy = x, y

    # Attentionを消す：Shiftキーを押しながら、マウスが動いたとき
    elif flags == cv2.EVENT_FLAG_SHIFTKEY and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(att_gray, (x, y), thickness, (0, 0, 0), -1)

    # Attentionをつける：Altキーを押しながら、マウスが動いたとき
    elif flags == cv2.EVENT_FLAG_ALTKEY and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(att_gray, (x, y), thickness, (220, 220, 220), -1)
        sx3, sy3 = x, y

# トラックバーの値を変更する度にRGBを出力する
def changePencil(pos):
    global palette_img, color, thickness
    r = cv2.getTrackbarPos("R", "palette")
    g = cv2.getTrackbarPos("G", "palette")
    b = cv2.getTrackbarPos("B", "palette")
    palette_img[:] = [b, g, r]
    color = (b, g, r)
    thickness = cv2.getTrackbarPos("T", "palette")

# パレット画像を作成
palette_img = np.zeros((200, 512, 3), np.uint8)

if __name__ == '__main__':
    img_count = 0
    data = []
    data1 = []
    data2 = []
    for file in tqdm(glob.glob('output/attention/*.png')):
        data.append(file)
        data.sort()

    for file in tqdm(glob.glob('output/attention_grayscale/*.png')):
        data1.append(file)
        data1.sort()

    for file in tqdm(glob.glob('output/raw/*.png')):
        data2.append(file)
        data2.sort()

    # ウィンドウの名前を設定
    cv2.namedWindow("Revise Attention map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Attention map", cv2.WINDOW_NORMAL)

    # パレットウィンドを生成
    cv2.namedWindow("palette")

    # マウス操作のコールバック関数の設定
    cv2.setMouseCallback("Revise Attention map", callback)

    # トラックバーのコールバック関数の設定
    cv2.createTrackbar("R", "palette", 0, 255, changePencil)
    cv2.createTrackbar("G", "palette", 0, 255, changePencil)
    cv2.createTrackbar("B", "palette", 0, 255, changePencil)
    cv2.createTrackbar("T", "palette", 1, 100, changePencil)

    end_flag = False
    for i in range(len(data)):
        # 画像を読み込む
        v_img = cv2.imread(data2[img_count])
        att = cv2.imread(data[img_count])
        att_gray = cv2.imread(data1[img_count])
        print("Revise image name:",data[img_count])
        if end_flag:
            break
        while(True):

            cv2.imshow("Revise Attention map", cv2.addWeighted(v_img, 1, att_gray, 1.0, 0))
            cv2.imshow("Attention map", att)
            cv2.imshow("palette", palette_img)
            k = cv2.waitKey(1)

            # Escキーを押すと終了
            if k == 27:
                end_flag = True
                break

            # sを押すと画像を保存
            elif k == ord("s"):
                edit_img = att_gray
                name = data[img_count].split('/')
                cv2.imwrite("revised_attention/resume/{}".format(name[2]), edit_img)
                edit_img = min_max(edit_img)
                edit_img = cv2.GaussianBlur(edit_img, (21,21), 100)
                cv2.imwrite("revised_attention/attention_map_grayscale/{}".format(name[2]), att_gray)
                attention_map = cv2.applyColorMap(edit_img.astype(np.uint8), cv2.COLORMAP_JET)
                jet_map = cv2.add(v_img.astype(np.uint8), attention_map)
                cv2.imwrite("revised_attention/attention_map/{}".format(name[2]), jet_map)
                cv2.imwrite("static/tmp.png", jet_map)
                cv2.imwrite("static/val_tmp.png", v_img)
                cv2.imwrite("static/org_tmp.png", att)
                cmd = "python3 server.py"
                print("Save compleate")

            elif k == ord("n"):
                img_count += 1
                break

            elif k == ord("b"):
                if img_count == len(data):
                    img_count = 0
                else:
                    img_count -= 1
                break

            elif k == ord('r'):
                print ("reset")
                att_gray = cv2.imread(data1[img_count])

            elif k == ord('q'):
                print ("revise resume")
                att_gray = cv2.imread("revised_attention/resume/{}".format(name[2]))

            elif k == ord('c'):
                edit_proc = subprocess.Popen(cmd.split())
                sleep(20)
                edit_proc.terminate()


