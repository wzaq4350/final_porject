# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import math
import threading
from sklearn.decomposition import PCA
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.messagebox import *


def str_to_int(st):
    return int(st.replace(" ", "").replace("\n", ""))


def str_to_float(st):
    return float(st.replace(" ", "").replace("\n", ""))


def pil_resize(w, h, pil_image):
    f1 = 1.0 * 500 / w
    f2 = 1.0 * 250 / h
    factor = min(f1, f2)
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


def super_resolution():
    global cv2_finish_img
    showinfo("注意", "請等待執行完成視窗跳出再執行下一動作")
    img = cv2_finish_img.copy()
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "ESPCN_x2.pb"
    sr.readModel(path)
    sr.setModel(
        "espcn", 2
    )  # set the model by passing the value and the upsampling ratio
    result_img = sr.upsample(img)  # upscale the input image
    showinfo("注意", "超級採樣完成")
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def cv2_resize():
    global cv2_finish_img
    # 原圖片長寬
    h, w = cv2_finish_img.shape[:2]
    factor = str_to_float(resize_text.get("1.0", "end"))
    # 縮放時，維持原圖片長寬比
    size = (int(w * factor), int(h * factor))
    if factor < 1:
        result_img = cv2.resize(cv2_finish_img, size, interpolation=cv2.INTER_LINEAR)
    else:
        result_img = cv2.resize(cv2_finish_img, size, interpolation=cv2.INTER_CUBIC)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()
    h_new, w_new = cv2_finish_img.shape[:2]
    showinfo("完成", f"原圖片解析度:{w}x{h}\n新的圖片解析度:{w_new}x{h_new}")


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def cv2_pic_preview():
    global cv2_finish_img
    img = cv2_finish_img.copy()
    img_open = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    reimg = pil_resize(img_open.width, img_open.height, img_open)
    preview_img = ImageTk.PhotoImage(reimg)
    process_image_label.config(image=preview_img)
    process_image_label.image = preview_img
    # 畫出 RGB 三種顏色的分佈圖
    color = ["b", "g", "r"]
    rgb_img = plt.figure(figsize=(12, 8))
    for i in range(3):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color[i])
        plt.xlim([0, 256])
    # 將matplotlib所繪出的直方圖格式轉換成PIL的格式做輸出
    rgb_histogram = fig2img(rgb_img)
    # 調整圖片大小
    rgb_histogram = rgb_histogram.resize(
        (int(rgb_histogram.width * 0.5), int(rgb_histogram.height * 0.5)),
        Image.ANTIALIAS,
    )
    # 顯示直方圖
    rgb_histogram = ImageTk.PhotoImage(rgb_histogram)
    histogram_label.config(image=rgb_histogram)
    histogram_label.image = rgb_histogram
    # 將matplotlib所繪出的直方圖清除，節省記憶體
    plt.close(rgb_img)


def choosepic():
    """選擇圖片"""
    global cv2_origin_img, cv2_finish_img
    # 選取檔案開啟位置
    path_ = askopenfilename()
    path.set(path_)
    img_open = Image.open(file_entry.get())
    # cv2 img
    open_path = file_entry.get()
    cv2_origin_img = cv2.imread(open_path)
    cv2_finish_img = cv2_origin_img.copy()
    # show the image
    reimg = pil_resize(img_open.width, img_open.height, img_open)
    img = ImageTk.PhotoImage(reimg)
    origin_image_label.config(image=img)
    origin_image_label.image = img
    cv2_pic_preview()


def savepic():
    """圖片儲存"""
    global cv2_finish_img
    # 選取檔案儲存位置
    path_ = asksaveasfilename()
    path.set(path_)
    save_path = file_entry.get()
    # 儲存
    cv2.imwrite(save_path, cv2_finish_img)


def modify_lightness():
    """亮度調整"""
    global cv2_finish_img
    # 圖像歸一化，且轉換為浮點型
    img = cv2_finish_img.astype(np.float64)
    img = img / 255.0
    lightness = (str_to_int(lightness_text.get("1.0", "end"))) / 100.0
    if lightness > 0:
        result_img = np.power(img, 1 / (1 + lightness))
    else:
        result_img = np.power(img, (1 - lightness) / 1)
    result_img = (result_img * 255.0).astype(np.uint8)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def modify_saturation():
    global cv2_finish_img
    # 圖像歸一化，且轉換為浮點型
    fImg = cv2_finish_img.astype(np.float32)
    fImg = fImg / 255.0
    # 顏色空間轉換至HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = hlsImg.copy()
    saturation = str_to_int(saturation_text.get("1.0", "end"))
    # 飽和度調整
    hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    # 調整後的值要介於 0~1，超過1的部分設定為1
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
    # 顏色空間轉換回BGR
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = (result_img * 255).astype(np.uint8)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def gaussian_filter():
    """高斯濾波器"""
    global cv2_finish_img
    showinfo("注意", "請等待執行完成視窗跳出再執行下一動作")
    # 濾波器矩陣為3x3
    K_size = 3
    sigma = 1.0
    img = np.asarray(np.uint8(cv2_finish_img))
    h, w, c = img.shape
    # Zero padding
    pad = K_size // 2
    out = np.zeros((h + pad * 2, w + pad * 2, c), dtype=np.float64)
    out[pad : pad + h, pad : pad + w] = img.copy().astype(np.float64)

    # prepare Kernel
    kernel = np.zeros((K_size, K_size), dtype=np.float64)
    for i in range(-pad, -pad + K_size):
        for j in range(-pad, -pad + K_size):
            kernel[j + pad, i + pad] = np.exp(-(i ** 2 + j ** 2) / (2 * (sigma ** 2)))
    kernel /= 2 * np.pi * sigma ** 2
    kernel /= kernel.sum()
    temp = out.copy()

    # filtering
    for i in range(h):
        for j in range(w):
            for k in range(c):
                out[pad + i, pad + j, k] = np.sum(
                    kernel * temp[i : i + K_size, j : j + K_size, k]
                )
    # 限制數值輸出範圍0~255
    out = np.clip(out, 0, 255)
    out = out[pad : pad + h, pad : pad + w].astype(np.uint8)
    showinfo("注意", "執行完成")
    cv2_finish_img = out.copy()
    cv2_pic_preview()


def gaussian_noise():
    """添加高斯噪點"""
    global cv2_finish_img
    sigma = str_to_float(gaussian_noise_text.get("1.0", "end"))
    img = cv2_finish_img.copy()
    # 將原圖像歸一化，使數值範圍介於0~1
    img = np.array(img / 255, dtype=np.float64)
    # 创建均值为0方差为var呈高斯分布的图像矩阵
    noise = np.random.normal(0, sigma, img.shape)
    # 將高斯雜訊加到圖片之中
    result_img = img + noise
    # 限制值的範圍為0~1
    result_img = np.clip(result_img, 0, 1)
    # 乘以255將數值範圍恢復到0~255
    result_img = (result_img * 255).astype(np.uint8)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def thread_it(func, *args):
    """開啟多線程"""
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()


def cv2_image_reduction():
    """復原圖片的調整"""
    global cv2_origin_img, cv2_finish_img
    cv2_finish_img = cv2_origin_img.copy()
    cv2_pic_preview()


def pca():
    """PCA圖片壓縮"""
    global cv2_finish_img
    img = cv2_finish_img.copy()
    # Splitting into channels
    imgB = img[:, :, 0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]
    pca_b = PCA(n_components=10)
    pca_b.fit(imgB)
    trans_pca_b = pca_b.transform(imgB)
    pca_g = PCA(n_components=10)
    pca_g.fit(imgG)
    trans_pca_g = pca_g.transform(imgG)
    pca_r = PCA(n_components=10)
    pca_r.fit(imgR)
    trans_pca_r = pca_r.transform(imgR)
    # 重建圖像並可視化
    b_arr = pca_b.inverse_transform(trans_pca_b)
    g_arr = pca_g.inverse_transform(trans_pca_g)
    r_arr = pca_r.inverse_transform(trans_pca_r)
    result_img = np.dstack((b_arr, g_arr, r_arr)).astype(np.uint8)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def modify_color_temperature():
    """調整圖片的色溫"""
    global cv2_finish_img
    img = cv2_finish_img.copy()
    # 分離出照片bgr通道
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    r_parameter = (str_to_int(r_channel_text.get("1.0", "end"))) / 100.0
    g_parameter = (str_to_int(g_channel_text.get("1.0", "end"))) / 100.0
    b_parameter = (str_to_int(b_channel_text.get("1.0", "end"))) / 100.0
    # 調整圖片的bgr的成分比例
    r = cv2.addWeighted(src1=r, alpha=1 + r_parameter, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=1 + g_parameter, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=1 + b_parameter, src2=0, beta=0, gamma=0)
    # 將各通道重建回圖片
    result_img = np.dstack((b, g, r)).astype(np.uint8)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def modify_contrast():
    global cv2_finish_img
    img = cv2_finish_img.copy()
    contrast = str_to_float(contrast_text.get("1.0", "end"))
    f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    result_img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def sharpen():
    """將圖片銳化，此方法為unsharp masking """
    global cv2_finish_img
    img = cv2_finish_img.copy()
    # unsharp masking
    sigma = str_to_int(sharpen_text.get("1.0", "end"))
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    # 以原圖 : 模糊圖片= 1.5 : -0.5 的比例進行疊加。
    result_img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def hightlight():
    global cv2_finish_img
    img = cv2_finish_img.copy()
    # 分離出照片bgr通道
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 將圖片高光部分選取出來
    parameter = (str_to_int(hightlight_text.get("1.0", "end"))) / 1000.0
    ret, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    b = cv2.addWeighted(src1=b, alpha=1, src2=mask, beta=parameter, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=1, src2=mask, beta=parameter, gamma=0)
    r = cv2.addWeighted(src1=r, alpha=1, src2=mask, beta=parameter, gamma=0)
    result_img = np.dstack((b, g, r)).astype(np.uint8)
    result_img = np.clip(result_img, 0, 255)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


def shadow():
    global cv2_finish_img
    img = cv2_finish_img.copy()
    # 分離出照片bgr通道
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 將圖片高光部分選取出來
    parameter = (str_to_int(shadow_text.get("1.0", "end"))) / 1000.0
    ret, mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)
    b = cv2.addWeighted(src1=b, alpha=1, src2=mask, beta=parameter, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=1, src2=mask, beta=parameter, gamma=0)
    r = cv2.addWeighted(src1=r, alpha=1, src2=mask, beta=parameter, gamma=0)
    result_img = np.dstack((b, g, r)).astype(np.uint8)
    result_img = np.clip(result_img, 0, 255)
    cv2_finish_img = result_img.copy()
    cv2_pic_preview()


root = tk.Tk()
root.geometry("1200x600")
root.title("圖片處理")
path = tk.StringVar()
file_entry = tk.Entry(root, state="readonly", text=path)

# label
origin_image_label = tk.Label(root)
origin_image_label.place(x=10, y=50)
process_image_label = tk.Label(root)
process_image_label.place(x=10, y=330)
histogram_label = tk.Label(root)
histogram_label.place(x=730, y=330)
lightness_label = tk.Label(root, text="亮度調整\n(-100%~100%)")
lightness_label.place(x=800, y=15)
saturation_label = tk.Label(root, text="飽和度調整\n(-100%~100%)")
saturation_label.place(x=800, y=65)
contrast_label = tk.Label(root, text="對比度調整\n(-20~20)")
contrast_label.place(x=812, y=115)
resize_label = tk.Label(root, text="解析度調整\n(調整倍率)")
resize_label.place(x=812, y=165)
gaussian_noise_label = tk.Label(root, text="添加噪點\n(0~1)")
gaussian_noise_label.place(x=817, y=215)
shadow_label = tk.Label(root, text="陰影\n(-100~100)")
shadow_label.place(x=810, y=265)
hightlight_label = tk.Label(root, text="高光\n(-100~100)")
hightlight_label.place(x=810, y=315)
color_temperature_label = tk.Label(root, text="RGB調整\n(-100~100)")
color_temperature_label.place(x=480, y=30)
sharpen_label = tk.Label(root, text="銳化\n(-100~100)")
sharpen_label.place(x=480, y=95)


# text
lightness_text = tk.Text(root, width=15, height=1)
lightness_text.place(x=900, y=20)
saturation_text = tk.Text(root, width=15, height=1)
saturation_text.place(x=900, y=70)
contrast_text = tk.Text(root, width=15, height=1)
contrast_text.place(x=900, y=120)
resize_text = tk.Text(root, width=15, height=1)
resize_text.place(x=900, y=170)
gaussian_noise_text = tk.Text(root, width=15, height=1)
gaussian_noise_text.place(x=900, y=220)
shadow_text = tk.Text(root, width=15, height=1)
shadow_text.place(x=900, y=270)
hightlight_text = tk.Text(root, width=15, height=1)
hightlight_text.place(x=900, y=320)
sharpen_text = tk.Text(root, width=15, height=1)
sharpen_text.place(x=580, y=100)
r_channel_text = tk.Text(root, width=15, height=1)
r_channel_text.place(x=580, y=20)
g_channel_text = tk.Text(root, width=15, height=1)
g_channel_text.place(x=580, y=40)
b_channel_text = tk.Text(root, width=15, height=1)
b_channel_text.place(x=580, y=60)


# button
tk.Button(root, text="選擇圖片", command=choosepic).grid(
    row=0, column=0, padx=10, pady=10, sticky=tk.W
)
tk.Button(root, text="儲存", command=savepic).grid(
    row=0, column=1, padx=10, pady=10, sticky=tk.W
)
tk.Button(root, text="高斯降噪", command=lambda: thread_it(gaussian_filter)).grid(
    row=0, column=2, padx=10, pady=10, sticky=tk.W
)
tk.Button(root, text="pca壓縮", command=lambda: thread_it(pca)).grid(
    row=0, column=3, padx=10, pady=10, sticky=tk.W
)
tk.Button(root, text="超級解析度(X2)", command=lambda: thread_it(super_resolution)).grid(
    row=0, column=3, padx=10, pady=10, sticky=tk.W
)
tk.Button(root, text="預設", command=lambda: thread_it(cv2_image_reduction)).grid(
    row=0, column=4, padx=10, pady=10, sticky=tk.W
)
tk.Button(root, text="輸入", command=lambda: thread_it(modify_lightness)).place(
    x=1050, y=15
)
tk.Button(root, text="輸入", command=lambda: thread_it(modify_saturation)).place(
    x=1050, y=65
)
tk.Button(root, text="輸入", command=lambda: thread_it(modify_contrast)).place(
    x=1050, y=115
)
tk.Button(root, text="輸入", command=lambda: thread_it(cv2_resize)).place(x=1050, y=165)
tk.Button(root, text="輸入", command=lambda: thread_it(gaussian_noise)).place(
    x=1050, y=215
)
tk.Button(root, text="輸入", command=lambda: thread_it(shadow)).place(x=1050, y=265)
tk.Button(root, text="輸入", command=lambda: thread_it(hightlight)).place(x=1050, y=315)
tk.Button(root, text="輸入", command=lambda: thread_it(sharpen)).place(x=710, y=95)
tk.Button(root, text="輸入", command=lambda: thread_it(modify_color_temperature)).place(
    x=710, y=35
)


root.mainloop()


# %%

