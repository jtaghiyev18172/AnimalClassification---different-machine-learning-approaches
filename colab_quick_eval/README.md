# Colab Quick Evaluation Package

This folder is a lightweight Google Colab demo for the course submission.

It contains:

- `customcnn_v2_colab_eval.ipynb` - Colab notebook that loads the saved scratch-CNN checkpoint and evaluates 100 test images.
- `test_images/` - 100-image subset from `split_v1` test data, organized by class.
- `test_manifest.csv` - image list with labels and original source paths.
- `weights/customcnn_v2_checkpoint.pt` - saved `CustomCNN v2` checkpoint.
- `weights/customcnn_v2_config.json` and `weights/customcnn_v2_metrics.json` - original run metadata.

Recommended Colab usage:

1. Upload this whole folder to `/content/colab_quick_eval/`.
2. Open `customcnn_v2_colab_eval.ipynb`.
3. Runtime -> Run all.

The notebook does not retrain the model. It only performs quick inference/evaluation.
