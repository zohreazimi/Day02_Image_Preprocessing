"""
Image Processing Week - Day 2
Smoothing Filters & Noise Reduction
پردازش تصویر - روز دوم
فیلترهای صاف‌کننده و حذف نویز

Author: Zohre Azimi
No dataset download needed -- uses a built-in scikit-image sample image.
نیازی به دانلود دیتاست نیست -- از یک تصویر نمونه‌ی آماده‌ی scikit-image استفاده می‌کنیم.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, util, filters, metrics

# -----------------------------------------------------------------------
# 1) Load a small, ready-to-use sample image (grayscale)
#    بارگذاری یک تصویر نمونه‌ی کوچک و آماده (سطح خاکستری)
# -----------------------------------------------------------------------
image = color.rgb2gray(data.astronaut())

# -----------------------------------------------------------------------
# 2) Add synthetic noise to simulate a real noisy image
#    اضافه کردن نویز مصنوعی برای شبیه‌سازی یک تصویر واقعی نویزی
#    Gaussian noise: common in real sensors / low-light imaging
#    نویز گوسی: در سنسورهای واقعی و تصویربرداری در نور کم رایج است
# -----------------------------------------------------------------------
noisy = util.random_noise(image, mode="gaussian", var=0.02)

# -----------------------------------------------------------------------
# 3) Apply three classic smoothing / denoising filters
#    اعمال سه فیلتر کلاسیک صاف‌کننده / حذف‌کننده‌ی نویز
# -----------------------------------------------------------------------
# a) Gaussian filter: weighted average, preserves edges better than mean
#    فیلتر گوسی: میانگین وزن‌دار، لبه‌ها را بهتر از فیلتر میانگین حفظ می‌کند
gaussian_denoised = filters.gaussian(noisy, sigma=1.2)

# b) Mean (uniform) filter: simple average over a local window
#    فیلتر میانگین: میانگین‌گیری ساده روی یک پنجره‌ی محلی
mean_denoised = filters.rank.mean(
    util.img_as_ubyte(noisy), footprint=np.ones((5, 5), dtype=bool)
)
mean_denoised = util.img_as_float(mean_denoised)

# c) Median filter: replaces each pixel with the median of its neighborhood;
#    very effective against salt-and-pepper-like noise while keeping edges sharp
#    فیلتر میانه: هر پیکسل را با میانه‌ی همسایگانش جایگزین می‌کند؛
#    در برابر نویز نمک‌و‌فلفل بسیار موثر است و لبه‌ها را تیز نگه می‌دارد
median_denoised = filters.median(noisy, footprint=np.ones((3, 3), dtype=bool))

# -----------------------------------------------------------------------
# 4) Compare quality using PSNR (Peak Signal-to-Noise Ratio)
#    مقایسه‌ی کیفیت با معیار PSNR (نسبت پیک سیگنال به نویز)
#    Higher PSNR -> result closer to the original clean image
#    PSNR بالاتر -> نتیجه نزدیک‌تر به تصویر اصلی و بدون نویز
# -----------------------------------------------------------------------
results = {
    "Noisy": noisy,
    "Gaussian filter": gaussian_denoised,
    "Mean filter": mean_denoised,
    "Median filter": median_denoised,
}

print("PSNR compared to the original image / مقایسه PSNR با تصویر اصلی:")
for name, img in results.items():
    psnr = metrics.peak_signal_noise_ratio(image, img)
    print(f"  {name:16s}: {psnr:.2f} dB")

# -----------------------------------------------------------------------
# 5) Visualize original, noisy, and all three filtered results
#    نمایش تصویر اصلی، نویزی و هر سه نتیجه‌ی فیلترشده
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.ravel()

panels = [
    ("Original / اصلی", image),
    ("Noisy (Gaussian noise) / نویزی", noisy),
    ("Gaussian filter", gaussian_denoised),
    ("Mean filter", mean_denoised),
    ("Median filter", median_denoised),
]

for ax, (title, img) in zip(axes, panels):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")

axes[-1].axis("off")  # empty panel

plt.tight_layout()
plt.savefig("day02_smoothing_denoising.png", dpi=150)
plt.show()

print("\nDone! / تمام شد!")
print("Figure saved as day02_smoothing_denoising.png")
