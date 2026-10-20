import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Dummy classes in case external engine modules are not present
try:
    from engine import MatrixFilterEngine
    import kernels
except ImportError:
    class MatrixFilterEngine:
        def __init__(self, img):
            self.img = np.array(img)
            self.height, self.width = self.img.shape[:2]

        def to_grayscale(self):
            if len(self.img.shape) == 3:
                return np.dot(self.img[..., :3], [0.2989, 0.5870, 0.1140])
            return self.img

        def invert(self):
            return 255 - self.img

        def convolve(self, kernel):
            # 1. Image ko grayscale mein convert karein
            gray = self.to_grayscale().astype(np.float32)

            # 2. Kernel ki height aur width nikalen
            kh, kw = kernel.shape
            pad_h = kh // 2
            pad_w = kw // 2

            # 3. Image boundary par zero padding lagayein
            padded_img = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

            # 4. Use NumPy sliding_window_view for vectorization
            # This creates a view of windows matching the kernel shape
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(padded_img, (kh, kw))

            # 5. Perform vectorized element-wise multiplication and summation over the window axes
            output = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

            # 6. Pixel values ko 0 aur 255 ke beech clip karein
            return np.clip(output, 0, 255).astype(np.uint8)

    class Kernels:
        SOBEL_V = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        SOBEL_H = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        BOX_BLUR_3X3 = np.ones((3, 3)) / 9.0
        SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        Emboss = np.array([[-2,1,0],[-1,1,1],[0,1,2]])
        Ridge = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    kernels = Kernels()

def run_project(image_path= 'gitimg.jpeg'):
    if image_path:
        display_original = Image.open(image_path).convert("RGB")
        engine = MatrixFilterEngine(np.array(display_original))
    else:
        canvas = np.zeros((240, 240, 3), dtype=np.uint8)
        canvas[:, :120, 0] = np.linspace(40, 200, 240)[:, None]
        canvas[:, :120, 2] = 160
        canvas[:, 120:, 1] = 255
        canvas[:, 120:, 0] = 100
        canvas[80:160, 80:160] = 255
        engine = MatrixFilterEngine(canvas)
        display_original = canvas

    print(f"Image Loaded: {engine.height}x{engine.width}")

    gray = engine.to_grayscale().astype(np.uint8)
    inverted = engine.invert()
    sobel_v = engine.convolve(kernels.SOBEL_V)
    sobel_h = engine.convolve(kernels.SOBEL_H)
    box_blur = engine.convolve(kernels.BOX_BLUR_3X3)
    sharpen = engine.convolve(kernels.SHARPEN)
    laplacian = engine.convolve(kernels.LAPLACIAN)
    emboss_effect = engine.convolve(kernels.Emboss)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(display_original)
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("Grayscale")

    axes[0, 2].imshow(inverted, cmap='gray' if len(inverted.shape) == 2 else None)
    axes[0, 2].set_title("Inverted")

    axes[0, 3].imshow(box_blur, cmap='gray')
    axes[0, 3].set_title("Box Blur (3x3)")

    axes[1, 0].imshow(sobel_v, cmap='gray')
    axes[1, 0].set_title("Sobel Vertical")

    axes[1, 1].imshow(sobel_h, cmap='gray')
    axes[1, 1].set_title("Sobel Horizontal")

    axes[1, 2].imshow(sharpen, cmap='gray')
    axes[1, 2].set_title("Sharpen")

    axes[1, 3].imshow(emboss_effect,cmap = 'gray')
    axes[1, 3].set_title("Laplacian")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("output_demo.png", dpi=600)
    plt.show()
    print("Execution complete!")

if __name__ == "__main__":
    run_project()