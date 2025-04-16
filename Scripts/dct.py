import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def apply_dct(block):
    """Apply 2D DCT to an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    """Apply 2D inverse DCT to an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def preprocess_watermark(watermark_path, shape):
    """Read and resize the watermark to match 8x8 block layout of the cover image."""
    watermark = cv2.imread(watermark_path, 0)
    watermark = cv2.resize(watermark, (shape[1] // 8, shape[0] // 8))
    _, binary = cv2.threshold(watermark, 128, 1, cv2.THRESH_BINARY) 
    return binary

def embed_watermark(cover, watermark_binary, alpha=10):
    """Embed the binary watermark into the DCT coefficients of the cover image."""
    h_blocks = cover.shape[0] // 8
    w_blocks = cover.shape[1] // 8
    watermarked = np.zeros_like(cover, dtype=np.float32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = cover[i*8:(i+1)*8, j*8:(j+1)*8]
            dct_block = apply_dct(block)

            # Embed watermark bit at mid-frequency position
            if watermark_binary[i, j] == 1:
                dct_block[4, 4] += alpha
            else:
                dct_block[4, 4] -= alpha

            idct_block = apply_idct(dct_block)
            watermarked[i*8:(i+1)*8, j*8:(j+1)*8] = idct_block

    return np.clip(watermarked, 0, 255).astype(np.uint8)

def show_images(title1, image1, title2, image2):
    """Display two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Show first image
    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')

    # Show second image
    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.show()


def show_image(title, image):
    #"""Display an image using matplotlib."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# === Main Process ===
cover_path = 'cover.jpg'
watermark_path = 'watermark.png'

cover = cv2.imread(cover_path, 0)
watermark_bin = preprocess_watermark(watermark_path, cover.shape)
watermarked_img = embed_watermark(cover, watermark_bin)

cv2.imwrite('watermarked.jpg', watermarked_img)

show_images("Cover Image", cover, "Watermarked Image", watermarked_img)
















