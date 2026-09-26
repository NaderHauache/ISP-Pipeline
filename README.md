# ISP-Pipeline

A minimal, educational **Image Signal Processing (ISP) pipeline** written in Python that converts **RAW camera images into sRGB images** and evaluates the result with **no-reference image quality metrics** (BRISQUE and ΔDoM).

This project was developed as part of my undergraduate thesis in Computer Engineering at the Federal University of Amazonas (UFAM, 2022):

> **Avaliação da Qualidade de Imagens Processadas em Pipelines de Processamento de Sinais por Métodos Cegos**
> *(Quality Assessment of Processed Images in Signal Processing Pipelines by Blind Methods)*
> Full text: http://riu.ufam.edu.br/handle/prefix/6383

---

## Pipeline overview

```
RAW (.ARW) ─► Black Level Correction ─► White Balance ─► Demosaicing ─► Color Space ─► Gamma ─► sRGB (.bmp)
              + Linearization          (camera gains)   (Malvar 2004)  Conversion     Correction
                                                                        (Camera → XYZ → sRGB)
                                                                              │
                                                                              ▼
                                                              IQ evaluation: BRISQUE + ΔDoM
```

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Loader** | Reads the RAW file with `rawpy` and loads the Bayer (CFA) data into a NumPy array, together with the camera metadata (CFA pattern, black/white levels, white balance gains, color matrix). |
| 2 | **Black Level Correction & Linearization** | Subtracts the per-channel black level and normalizes the signal to the [0, 1] range using the sensor white level. |
| 3 | **White Balance** | Applies the white balance gains stored in the camera metadata, normalized to the green channel. |
| 4 | **CFA visualization** *(optional)* | Saves a color-coded image of the Bayer mosaic, showing which pixel belongs to each color channel. |
| 5 | **Demosaicing** | Reconstructs the full RGB image using the Malvar (2004) method from the `colour-demosaicing` library. |
| 6 | **Color Space Conversion** | Converts camera RGB to sRGB through the XYZ color space, using the camera's color matrix from the metadata. |
| 7 | **Gamma Correction** | Applies the standard sRGB transfer function. |

After processing, the script can evaluate the final image with two **no-reference (blind)** metrics:

- **BRISQUE** (*Blind/Referenceless Image Spatial Quality Evaluator*), via the [`image-quality`](https://github.com/ocampor/image-quality) library. Score range 0–100; lower usually means better perceived quality.
- **ΔDoM** (*Difference of Differences in a Median-filtered image*), a sharpness estimator, via [`pydom`](https://github.com/umang-singhal/pydom). Score range 0–√2; higher means sharper.

## Results

Scores obtained for the 8 sample images used in the thesis (from [`iq_assesment.csv`](iq_assesment.csv)):

| Image | BRISQUE | ΔDoM (sharpness) |
|-------|--------:|-----------------:|
| dog | 33.16 | 0.806 |
| corvette | 42.45 | 0.713 |
| dj | 26.16 | 1.092 |
| mercedes | 29.41 | 0.926 |
| motorcycle | 30.67 | 0.947 |
| sea-beach | 27.16 | 1.003 |
| river | 23.89 | 0.992 |
| lake | 43.51 | 0.958 |

In the thesis, the analysis of the pipeline stages showed that **linearization** and **white balance** had the greatest impact on the final quality indices. Based on that, complementary processing stages were proposed to improve images acquired by smartphones. See the full text for the complete methodology and discussion.

## Requirements

- Python 3
- [`rawpy`](https://github.com/letmaik/rawpy)
- [`numpy`](https://numpy.org/)
- [`imageio`](https://imageio.readthedocs.io/)
- [`colour-demosaicing`](https://github.com/colour-science/colour-demosaicing)
- [`image-quality`](https://github.com/ocampor/image-quality) *(BRISQUE; only needed for evaluation)*
- [`pydom`](https://github.com/umang-singhal/pydom) *(ΔDoM; only needed for evaluation; see its repository for installation)*

## Usage

1. Place your RAW file inside `RAW_Images/`.
2. Create an `Output/` folder in the project root (processed images are saved there).
3. Edit the configuration variables at the top of `ISP Pipeline.py`:

   ```python
   path = "RAW_Images/"   # input folder
   name_pic = "dog"       # file name without extension
   ext = ".ARW"           # RAW extension
   saveOp = False         # True = also save every intermediate stage (TIFF/BMP)
   EvalOp = True          # True = compute BRISQUE and ΔDoM on the final image
   ```

4. Run:

   ```bash
   python "ISP Pipeline.py"
   ```

The final image is saved as `Output/<name>_sRGB.bmp`, and the metric scores are printed to the terminal.

> **Note:** although `rawpy` supports many RAW formats, this pipeline was tested only with Sony `.ARW` files. Other formats and CFA layouts may require adjustments.

## Limitations and future work

This is an educational, first version (1.0-alpha). Possible next steps:

- Own implementation of demosaicing (instead of a library)
- Lens Shading Correction
- Noise reduction
- Sharpening
- Contrast and brightness adjustment
- Auto White Balance algorithm (instead of the metadata gains)
- Support for other CFA layouts (e.g., Quad Bayer) and monochrome sensors
- Performance improvements (e.g., a port to a compiled language such as Rust)

## Citation

If this work is useful to you, please cite the thesis:

```bibtex
@misc{hauache2022isp,
  author       = {Hauache, Nader Moraes},
  title        = {Avaliação da Qualidade de Imagens Processadas em Pipelines de Processamento de Sinais por Métodos Cegos},
  howpublished = {Trabalho de Conclusão de Curso (Engenharia da Computação), Universidade Federal do Amazonas},
  year         = {2022},
  url          = {http://riu.ufam.edu.br/handle/prefix/6383}
}
```

## Acknowledgments

Sample RAW images provided by [Signature Edits – Free RAW Photos](https://www.signatureedits.com/free-raw-photos/).

## License

Licensed under the [GNU General Public License v3.0](LICENSE).
