# 1. MRI Reconstruction Problem

**MRI Acquisition**

- Scanners do not measure images directly (k-space)
- Image is obtained via $I = \mathcal{F}^{-1}(k)$ where $\mathcal{F}^{-1}$ is the inverse Fourier, transforming from frequency (k) domain to image domain which is comprehensible.

**Main Challenge**

Scans are slow due to necessity to sample k-space densely. (cover more angles, obtain high-quality)

To speed up scans:
- Sample a subset of k-space (undersampling)
- Reconstruct missing values to recover HQ image.

# 2. Acceleration

Acceleration factor $R$ determines how much k-space is skipped. (sample every $R$ line)

e.g.
| Acceleration | Meaning                    |
| ------------ | -------------------------  |
| R = 4        | 25% of total k-space       |
| R = 8        | 12.5% of total k-space     |

Raise $R$:
- Faster scan
- Aliasing issue (nyquist)
- More artifacts

Overall, trade-off between **fast** and **high-quality**.


# 3. Sampling Considerations

Center of k-space contains important information (low freq)
- structure of image
- contrast of image

**center_fraction**: So a ratio of the center is sampled fully, rest is sampled with chosen method.
e.g.
|---- random ----|==== center ====|---- random ----|


# 4. Reconstruction Methods

## 4.1 Zero-Filled Reconstruction

Simplest. Steps are:
1. Fill missing k-space with zeroes
2. Apply fast inverse fourier transform `ifft2`

Results:
- Fast, almost no computation
- Blurry images, too much information lost
- Strong artifacts, low PSNR/SSIM


## 4.2 ESPIRiT



## 4.3 U-Net (pretrained baseline)

Encoder - Decoder network to learn to map zero-filled images to reconstructed HQ images.
Advantages:
- Removing artifacts that result from aliasing
- Better overall image quality (PSNR/SSIM)
- Removes the processes in-between. Could learn more, different reconstructions (e.g. MRI + CT fusion)


# 5. Metrics

## 5.1 NMSE

## 5.2 PSNR

## 5.3 SSIM
