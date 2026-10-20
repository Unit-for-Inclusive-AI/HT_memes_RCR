# HT Meme Dataset: Multi-Language Statistics & Data Profile

This document outlines the dataset statistics, label distributions, split compositions, and image metadata for the Anti-LGBT+ multimodal meme dataset across three languages: **Chinese**, **English**, and **Hindi**.

---

## 1. Executive Summary & Split Ratios

The raw dataset is stored under [`data/raw/`](../data/raw/). Every language follows an exact ~**80% Train / 20% Test** split. All images have verified labels and valid integrity (0 corrupted images, 0 missing files).

| Language | Train Count | Test Count | Total Memes | Train Split % | Test Split % | Image Formats |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chinese** | 956 | 239 | **1,195** | 80.00% | 20.00% | `.jpg`, `.jpeg`, `.gif`, `.webp`, `.png` |
| **English** | 560 | 141 | **701** | 79.89% | 20.11% | `.jpg`, `.jpeg` |
| **Hindi** | 798 | 200 | **998** | 79.96% | 20.04% | `.jpg`, `.jpeg` |
| **Total** | **2,314** | **580** | **2,894** | **79.96%** | **20.04%** | — |

---

## 2. Language-Specific Distributions

### 2.1 Chinese (`data/raw/Chinese/`)

- **Labels file (Train)**: [`data/raw/Chinese/Train/Train_labels.xlsx`](../data/raw/Chinese/Train/Train_labels.xlsx) (`id`, `label`)
- **Labels file (Test)**: [`data/raw/Chinese/Test_labels.xlsx`](../data/raw/Chinese/Test_labels.xlsx) (`image_id`, `class`)
- **Unlabeled Test**: [`data/raw/Chinese/Test/Test_without_labels.xlsx`](../data/raw/Chinese/Test/Test_without_labels.xlsx)
- **Image directories**: 
  - Train: [`data/raw/Chinese/Train/Train_images/`](../data/raw/Chinese/Train/Train_images/) (956 files)
  - Test: [`data/raw/Chinese/Test/Test_images/`](../data/raw/Chinese/Test/Test_images/) (239 files)

#### Class Distribution

| Class Name | Train Count | Train % | Test Count | Test % | Total Count | Total % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Homophobic` | 705 | 73.74% | 176 | 73.64% | 881 | 73.72% |
| `Non_Anti_LGBT` | 196 | 20.50% | 49 | 20.50% | 245 | 20.50% |
| `Transphobic` | 55 | 5.75% | 14 | 5.86% | 69 | 5.77% |
| **Total** | **956** | **100.00%** | **239** | **100.00%** | **1,195** | **100.00%** |

#### Binary Aggregation (Anti-LGBT vs. Non-Anti-LGBT)
- **Anti-LGBT** (`Homophobic` + `Transphobic`): 950 (79.50%) — *Train: 760 (79.50%), Test: 190 (79.50%)*
- **Non-Anti-LGBT**: 245 (20.50%) — *Train: 196 (20.50%), Test: 49 (20.50%)*

#### Image Metadata
- **Train Extensions**: `.jpg` (769), `.jpeg` (117), `.gif` (37), `.webp` (20), `.png` (13)
- **Test Extensions**: `.jpg` (194), `.jpeg` (27), `.gif` (8), `.webp` (7), `.png` (3)
- **Train Resolution**: Median: 728×750, Mean: 759×762 (Min: 86×86, Max: 1947×2340)
- **Test Resolution**: Median: 720×752, Mean: 798×791 (Min: 150×132, Max: 3847×2067)
- **Color Modes**: Train: RGB (911), Palette/P (42), RGBA (3) | Test: RGB (229), Palette/P (10)

---

### 2.2 English (`data/raw/English/`)

- **Labels file (Train)**: [`data/raw/English/Train/Train_labels.xlsx`](../data/raw/English/Train/Train_labels.xlsx) (`id`, `label`)
- **Labels file (Test)**: [`data/raw/English/Test_labels.xlsx`](../data/raw/English/Test_labels.xlsx) (`id`, `label`)
- **Unlabeled Test**: [`data/raw/English/Test/Test_without_labels.xlsx`](../data/raw/English/Test/Test_without_labels.xlsx)
- **Image directories**: 
  - Train: [`data/raw/English/Train/Train_images/`](../data/raw/English/Train/Train_images/) (560 files)
  - Test: [`data/raw/English/Test/Test_images/`](../data/raw/English/Test/Test_images/) (141 files)

#### Class Distribution

> **Note on Naming**: English uses noun labels (`Homophobia`, `Transphobia`) instead of adjective labels (`Homophobic`, `Transphobic`).

| Class Name | Train Count | Train % | Test Count | Test % | Total Count | Total % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Non_Anti_LGBT` | 240 | 42.86% | 60 | 42.55% | 300 | 42.80% |
| `Transphobia` | 160 | 28.57% | 41 | 29.08% | 201 | 28.67% |
| `Homophobia` | 160 | 28.57% | 40 | 28.37% | 200 | 28.53% |
| **Total** | **560** | **100.00%** | **141** | **100.00%** | **701** | **100.00%** |

#### Binary Aggregation (Anti-LGBT vs. Non-Anti-LGBT)
- **Anti-LGBT** (`Homophobia` + `Transphobia`): 401 (57.20%) — *Train: 320 (57.14%), Test: 81 (57.45%)*
- **Non-Anti-LGBT**: 300 (42.80%) — *Train: 240 (42.86%), Test: 60 (42.55%)*

#### Image Metadata
- **Train Extensions**: `.jpg` (555), `.jpeg` (5)
- **Test Extensions**: `.jpg` (141)
- **Train Resolution**: Median: 640×582, Mean: 674×700 (Min: 200×112, Max: 3742×6679)
- **Test Resolution**: Median: 640×568, Mean: 667×642 (Min: 150×150, Max: 2827×2827)
- **Color Modes**: Train: RGB (545), RGBA (9), Palette/P (6) | Test: RGB (139), RGBA (1), Palette/P (1)

---

### 2.3 Hindi (`data/raw/Hindi/`)

- **Labels file (Train)**: [`data/raw/Hindi/Train/Train_labels.xlsx`](../data/raw/Hindi/Train/Train_labels.xlsx) (`id`, `label`)
- **Labels file (Test)**: [`data/raw/Hindi/Test_labels.xlsx`](../data/raw/Hindi/Test_labels.xlsx) (`id`, `label`)
- **Unlabeled Test**: [`data/raw/Hindi/Test/Test_without_labels.xlsx`](../data/raw/Hindi/Test/Test_without_labels.xlsx)
- **Image directories**:
  - Train: [`data/raw/Hindi/Train/Train_Images/`](../data/raw/Hindi/Train/Train_Images/) (798 files — *capitalized "Images"*)
  - Test: [`data/raw/Hindi/Test/Test_Images/`](../data/raw/Hindi/Test/Test_Images/) (200 files — *capitalized "Images"*)

#### Class Distribution

| Class Name | Train Count | Train % | Test Count | Test % | Total Count | Total % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Homophobic` | 366 | 45.86% | 64 | 32.00% | 430 | 43.09% |
| `Transphobic` | 274 | 34.34% | 96 | 48.00% | 370 | 37.07% |
| `Non_Anti_LGBT` | 158 | 19.80% | 40 | 20.00% | 198 | 19.84% |
| **Total** | **798** | **100.00%** | **200** | **100.00%** | **998** | **100.00%** |

#### Binary Aggregation (Anti-LGBT vs. Non-Anti-LGBT)
- **Anti-LGBT** (`Homophobic` + `Transphobic`): 800 (80.16%) — *Train: 640 (80.20%), Test: 160 (80.00%)*
- **Non-Anti-LGBT**: 198 (19.84%) — *Train: 158 (19.80%), Test: 40 (20.00%)*

#### Image Metadata
- **Train Extensions**: `.jpg` (567), `.jpeg` (231)
- **Test Extensions**: `.jpg` (158), `.jpeg` (42)
- **Train Resolution**: Median: 640×516, Mean: 693×657 (Min: 165×175, Max: 4096×4096)
- **Test Resolution**: Median: 601×518, Mean: 706×675 (Min: 168×183, Max: 2880×2880)
- **Color Modes**: Train: RGB (798) | Test: RGB (200)

---

## 3. Engineering & Preprocessing Caveats

When loading this dataset in Python pipelines or model scripts:

1. **Class Label Harmonization**:
   - Chinese and Hindi use `Homophobic` and `Transphobic`.
   - English uses `Homophobia` and `Transphobia`.
   - If training a multilingual model simultaneously, map English labels to the adjective forms (`Homophobia -> Homophobic`, `Transphobia -> Transphobic`) or vice versa to align target class indices.
2. **Column Name Differences**:
   - Chinese Test Excel file ([`Test_labels.xlsx`](../data/raw/Chinese/Test_labels.xlsx)) uses headers `['image_id', 'class']`.
   - All other Excel files use `['id', 'label']`.
   - Excel reader scripts should normalize columns by positional index (`col 0 = id`, `col 1 = label`) or explicit mapping.
3. **Directory Path Casing**:
   - Hindi directories use capital `I`: `Train_Images` and `Test_Images`.
   - Chinese and English use lower-case `i`: `Train_images` and `Test_images`.
   - Use case-insensitive checks or check both path variants in scripts.
4. **Blank Rows in Test Files**:
   - Chinese and English `Test_labels.xlsx` contain 1000 rows in Excel, but only 239 (Chinese) and 141 (English) contain data. Always call `.dropna(how='all')` or filter out empty rows when reading with `pandas` or `openpyxl`.
5. **Non-RGB Modes and Animated Formats**:
   - Chinese and English contain palette-based images (`mode P`) and RGBA images, as well as animated/multi-frame GIFs (`.gif`) and `.webp`.
   - Image loading code should explicitly call `image.convert('RGB')` before passing tensors to vision-language backends (e.g. Qwen2.5-VL or CLIP).

