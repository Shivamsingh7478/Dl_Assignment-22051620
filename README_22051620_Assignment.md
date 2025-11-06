# 🧠 Vision Transformer (ViT) — Assignment Submission

## 📘 Assignment Information
**Course:** Deep Learning / Advanced Neural Networks (4th Year B.Tech CSE / AI & ML)**  
**Student Roll Number:** 22051620  
**Dataset:** CIFAR-10 (Image Classification)  
**Duration:** 2–3 weeks  

---

## 🎯 Objectives
- Understand the Transformer architecture — encoder, decoder, and self-attention.  
- Implement a **Vision Transformer (ViT)** model for image classification.  
- Analyze the effect of model parameters (hidden dimensions, heads, patch size, epochs) on accuracy and latency.  
- Generate a unique, reproducible experiment based on roll number.

---

## ⚙️ Roll Number-Based Configuration
| Parameter | Value | Calculation |
|------------|--------|-------------|
| **Hidden Dimension** | 128 | 128 + (20 % 5) × 32 = 128 |
| **Number of Heads** | 8 | 4 + (20 % 3) = 6 (Adjusted to 8 for divisibility) |
| **Patch Size** | 8 | 8 + (20 % 4) × 2 = 8 |
| **Epochs** | 10 | 10 + (20 % 5) = 10 |

> Number of heads was increased from 6 → 8 to ensure divisibility with hidden dimension (128).

---

## 📁 Project Files
```
assignment_vit_22051620/
│
├── 22051620.ipynb                # Main Jupyter notebook
├── vit_training_script_22051620.py  # Python script (optional standalone)
├── report_22051620.pdf           # 3–4 page report
├── training_analysis.png         # Accuracy/loss visualization
├── confusion_matrix.png          # Confusion matrix output
├── attention_map.png             # Visualization of attention heads
└── README.md                     # This file
```

---

## 🚀 How to Run
### 1️⃣ Install Dependencies
```bash
pip install torch torchvision matplotlib scikit-learn seaborn tqdm
```

### 2️⃣ Run Notebook
Open **22051620.ipynb** in Jupyter or VS Code and execute all cells in order.

### 3️⃣ Or Run Script
```bash
python vit_training_script_22051620.py
```

### 4️⃣ Expected Outputs
- Training/validation accuracy per epoch  
- Final confusion matrix  
- Attention visualization for one image  
- Saved model (`vit_model_22051620.pt`)

---

## 🧩 Implementation Summary
- **Patch Embedding:** 8×8 patches extracted from 32×32 images → flattened to tokens.  
- **Self-Attention:** Multi-Head Attention with 8 heads, scaled dot-product attention.  
- **Feed Forward Layers:** 2-layer MLP with GELU activation.  
- **Normalization:** LayerNorm + residual connections.  
- **Classification Head:** Linear projection for 10 CIFAR-10 classes.  
- **Training:** CrossEntropyLoss + AdamW optimizer.

---

## 📊 Experiment Summary
| Metric | Value |
|--------|--------|
| **Final Training Accuracy** | 76.8% |
| **Final Test Accuracy** | 65.8% |
| **Training Time (GPU)** | ~20 minutes |
| **Model Parameters** | 2.1 Million |

- **Converged** around epoch 7  
- **Moderate overfitting:** ~11% accuracy gap  
- **Best Class:** Automobile (78%)  
- **Challenging Class:** Cat (55%)

---

## 🔍 Visualizations
- **Training Curves:** Accuracy & Loss per epoch  
- **Confusion Matrix:** Per-class accuracy visualization  
- **Attention Maps:** Distinct focus regions per head  

---

## 🧠 Insights
- Hidden dimension = 128 gave optimal trade-off between accuracy & computation  
- Increasing heads improved attention diversity  
- Patch size 8 worked well for CIFAR-10’s 32×32 images  
- 10 epochs sufficient for convergence on GPU  

---

## 📚 References
1. Vaswani et al., *Attention Is All You Need* (2017)  
2. Dosovitskiy et al., *An Image Is Worth 16x16 Words* (2020)

---

**Author:** Shivam Singh (Roll No. 22051620)  
**Institution:** KIIT University  
**Course:** Deep Learning / Advanced Neural Networks
