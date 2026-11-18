# Seg Dataset Lab

Local COCO segmentation dataset analysis and inspection. Phase 1 provides dataset scanning, per-image or consolidated COCO JSON, optional split indexes, filename grouping, object size filters, image/GT preview and HTML/CSV export. Augmentation and copy-paste are planned for a later phase.

## Windows setup

Install Python and [uv](https://docs.astral.sh/uv/), then in PowerShell:

```powershell
git clone https://github.com/vvclin-git/seg_dataset_lab.git
cd seg_dataset_lab
uv sync
uv run python seg_dataset_lab.py
```

Choose an image root and a COCO JSON file or a folder containing per-image JSONs. Optionally select train/val/test indexes. Index lines may contain an image path alone or an image path followed by a comma and label path. Matching uses relative path first, then a unique basename. Unmatched entries and duplicate/conflicting assignments appear in the report. GT coordinates use the dimensions recorded in JSON, scaled to the working grid, even when image dimensions differ.

Regex rules use Python named groups such as `scene`, `camera`, `run`, `segment`, `frame`; separate multiple expressions with `;;`. The first match is used and overlapping matches are reported. Choose a category ID, size bin, scene and split to filter. Click an object for its preview, use the wheel to zoom and drag to pan. Save/open project stores paths and settings; report export produces HTML and CSV files. Original images and labels are read only.

Working grid uses direct resize, not letterboxing or cropping. Compare this assumption with your model's preprocessing. `objects.csv` contains stable IDs for lookup for a given image relative path and annotation ID. Polygon masks are rasterized by Pillow, which can differ at boundaries from other COCO rasterizers. RLE requires pycocotools. Missing GT is distinct from an annotated image with no target. Inspect the issues section for ambiguous or unavailable data.

## Scanning, inspection, and statistics

The progress bar shows the current scan phase. File discovery and split-index reading use an indeterminate bar; GT processing and grouping show completed counts and a phase percentage. The percentage refers to the current phase, not the entire scan. No images are copied during scanning.

The left pane is split into an image list and a GT list for the selected image. Images with no GT or no objects remain visible when the bin filter is `All`. Select a GT row to view its bounding box and mask. Rows outside the selected category or bin are greyed; scene and split filters restrict the image list.

Choose **Statistics** to see a histogram of bbox diagonal lengths on the working grid. Enter strictly increasing positive bin edges, such as `8,16,32`, and press **Apply bins**. Clicking a bar selects that bin in the main view. The chart respects the current category, scene and split; it shows all bins regardless of the selected bin. Export uses the applied bin edges and the active filters. The optional `8–<16 and height <4` filter always uses those fixed thresholds. Save the project to retain the applied bin edges.

**Export report** defaults to HTML and CSV only. To include copies of original images containing currently filtered objects, check the image-copy option; copies retain their dataset-relative paths under `images/`. Choose a report folder outside the dataset root. A report already exported to the chosen folder may have old images from an earlier export; unchecking the option does not delete them. Use a fresh folder for a report without image files.
