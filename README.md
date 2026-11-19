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

### Object-to-background brightness difference

Enable **Measure brightness on scan** before selecting **Scan**. Set **Ring (px)**, **Min object px**, and **Min background px** before scanning. Rescan after changing measurement settings. Click the **ΔY** heading in the GT list to sort, or select an object to see its two brightness medians and sample counts. The sign is background minus object. Positive values mean the object is darker than its surroundings.

The measurement uses the configured working grid. RGB uses bilinear direct resize and the instance mask uses nearest-neighbor resize. Pixel brightness is `Y = 0.2126R + 0.7152G + 0.0722B` on 0–255 RGB values. The background is the ring outside the instance mask, clipped to the image. Available masks of the same class and categories named Boat or Own Boat are excluded from this background sample. The defaults are 12 px for the radius and 32/64 minimum object/background pixels. Missing or empty instance masks, crowd annotations, and uncertain GT are ineligible for ΔY.

After a measured scan, choose **Signed**, **Positive**, **Negative**, or **Absolute**, enter inclusive bounds, and click **Apply ΔY range**. A blank bound uses the edge of the corresponding range. Positive and Negative exclude zero, while Signed and Absolute can include it. The same object must meet category, size, and brightness. Unknown scores cannot satisfy a brightness range. **All** requires each selected-category object to pass. **Mode** uses the existing most frequent size bin and requires a brightness hit in that bin. **Without category** ignores size and brightness.

**Statistics** summarizes measured and unknown counts, positive/zero/negative counts, and the ΔY distribution. Install the optional plots extra with `uv sync --extra plots` in an environment with Matplotlib support to see the Matplotlib histogram and size × ΔY scatter plot. The built-in chart displays the distribution when Matplotlib is absent.

The HTML report links the CSVs and local source images, summarizes split × scene, size distribution, and brightness, and lists examples nearest zero. All selected-category objects in admitted images contribute to the statistics and export. `objects.csv` includes both medians, ΔY, available pixel counts, and `brightness_status`. `images.csv` includes effective Regex status and the individual rule hits. When explicit Scene rules are configured, report coverage uses those rules. Grouping rule hits remain available for auditing the named groups.

The progress bar shows the current scan phase. File discovery and split-index reading use an indeterminate bar; GT processing and grouping show completed counts and a phase percentage. The percentage refers to the current phase, not the entire scan. No images are copied during scanning.

The left pane is split into an image list and a GT list for the selected image. Images with no GT or no objects remain visible when both category and bin are `All`. Select a GT row to view its bounding box and mask. Rows outside the selected category or bin are greyed; scene and split filters restrict the image list.

Choose **Statistics** to see a histogram of bbox diagonal lengths on the working grid. Enter strictly increasing positive bin edges, such as `8,16,32`, and press **Apply bins**. Clicking a bar selects that bin in the main view. The chart shows all objects of the selected category in the filtered images, across all size bins. The selected bin controls which images qualify, not which objects from those images are counted. Export uses the applied bin edges and the active filters. The optional `8–<16 and height <4` filter always uses those fixed thresholds. Save the project to retain the applied bin edges.

**Export report** defaults to HTML and CSV only. To include copies of the currently filtered original images, check the image-copy option; copies retain their dataset-relative paths under `images/`. Choose a report folder outside the dataset root. A report already exported to the chosen folder may have old images from an earlier export; unchecking the option does not delete them. Use a fresh folder for a report without image files.


## Image filtering and scene rules

Category, bin, scene and split filter the image list together. A category/bin match means at least one GT object meets both conditions. Use category `All` and bin `All` to include images without GT or objects. Statistics and reports count all objects of the selected category in those filtered images, including objects outside the selected bin. The report image denominator is the filtered image count. Click image-list column headers to sort; click again to reverse.

Select **Scenes** to enter one rule per line as `name | regex`. Separate the name from the expression with the first `|`; subsequent `|` characters remain part of the regex. A blank name assigns A, B, C by rule position. The first matching rule wins; Preview lists the assigned scene, unmatched images, and all rule hits. Regex applies to the filename or relative path according to the main selector. Without rules, existing `scene` groups extracted by the main grouping regex receive A, B, C in sorted order; files with no extracted scene are unmatched. Scene rules are retained in the project.


## Image match modes

The **Mode** selector controls how Category and Bin select images. `Any` requires one matching object. `All` requires at least one selected-category object and every one must match the bin. `Mode` selects images when the chosen bin has the largest object count among bins; ties qualify and are marked `tie`. The comparison considers only the selected category. `Without category` selects annotated images with none of the chosen category; missing-GT images are excluded. Select category `All` to consider all annotated categories. With category and bin both `All`, `Any` shows all images, including missing GT; `All` and `Mode` require at least one annotation. `Without category` ignores the bin, including when shading GT rows; rows of other categories remain grey.

The image-list `match` column shows matching/total selected-category objects (`—` when GT is missing). When the special `8–<16 and height <4` bin is used with Mode, matching objects form one bucket; other objects are counted in their ordinary diagonal bins, after removing the special-bin objects. Statistics remain all selected-category objects in filtered images, and the HTML report records the filter settings. Project files retain the match mode.


## Preview and layout

Drag either raised divider between the image list, GT list and preview to resize the panes. Click a GT column heading to sort ascending or descending; `size` sorts by numeric bbox diagonal and `area` by numeric working-grid mask area. The selected object remains selected when sorting.

`GT overlay` toggles mask display. With `All masks` checked, all valid masks in the current image are shown in category colors; the selected object is highlighted in yellow. Uncheck `All masks` to display only the selected object's mask. Click a displayed mask to select its GT row. Where masks overlap, the smallest mask is selected first, which helps select a small object over a larger background. Bbox-only objects have no clickable mask; select those from the GT list. Drag the preview to pan and use the mouse wheel to zoom. Masks are cached only in memory for the current image.
# GT status and exported image inventory

The image list shows `valid`, `missing`, `invalid`, `duplicate`, or `duplicate + invalid`. A question mark on the match count marks an uncertain count. `Without category` includes only valid GT with no object of the selected category; missing, invalid, and duplicate GT are never treated as confirmed absence. Other filters can show partial objects from uncertain GT for inspection. A duplicate uses the first loaded record; it does not merge records. Invalid annotations can be skipped, so these counts are provisional. Orphan annotations make the associated document's traceable images invalid because their intended image is unknown.

Every export includes `images.csv`, one row per selected image with split, scene, GT status, all traceable candidate source paths (JSON array, including repeated records from one file), and the count of loaded objects in the selected category (all sizes, as in `objects.csv`). This count can be incomplete for invalid or duplicate GT. The HTML target-presence totals count valid GT only. Malformed or unreadable files that cannot identify an image are reported under Issues but cannot be associated with an image; a missing status alone does not prove no such file exists.
