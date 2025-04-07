# Custom BM3D Denoising Method Implementation

**BM3D (Block-Matching and 3D Filtering)** introduced in **2007** in paper
**"Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering"**

This image denoising algorithm has long been considered the gold standard in
the field

Its innovative approach of **grouping similar** image patches and applying
**collaborative 3D filtering** has set a benchmark for balancing noise reduction
and detail preservation

## How BM3D Works 🛠️

1. **Block Matching**: Groups similar image patches (blocks) into 3D groups based on their
   similarity
2. **3D Transform and Collaborative Filtering**: Applies 3D transform to the
   grouped blocks, followed by a filter to remove noise

## Installation ⚙️

1. **Clone this repository**:
```bash
    git clone --depth=1
    https://github.com/IBIBENUDM/bm3d.git
```
2. **Install Package**:
```bash
    cd bm3d
    pip install .
```
## Demonstration 🖼️

Check out the `examples/` folder for a file named `demo.py`

This script demonstrates how to use the BM3D denoising algorithm and
visualize the results

## Branches 🌳

- `main` - Stable version of the project

- `dev` - Development of new features, improvements and bug fixes

- `bayessian-optimization` - Finding optimal hard threshold value

- `unet` - U-Net realization for image denoising
## Roadmap 🗺️

### Current Status ✅
- [x] **Basic Two-Stage Implementation**
- [x] **Optimizations from the Original Paper**
- [x] **Multithreading Support**
- [x] **Parameter Optimization via bayessian optimization**
- [x] **Implement U-Net Model** 
- [x] **Implement Numba**

### Future Goals 🚀
- [ ] **Optimize Key Functions in C++**
- [ ] **Benchmark Against Other Methods**

## License 📜

This project is licensed under the MIT License. For more details check out
the [**LICENSE**](../LICENSE) file
