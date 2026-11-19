"""Local COCO dataset inspector. Original files are never modified."""
import csv
import hashlib
import html
import json
import math
import re
import threading
import queue
import shutil
import time
from urllib.parse import quote
from collections import Counter, defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
BINS = ('<8', '8–<16', '16–<32', '≥32')
RING_RADIUS = 12
MIN_OBJECT_PIXELS = 32
MIN_BACKGROUND_PIXELS = 64
BRIGHTNESS_FIELDS = ('object_y', 'background_y', 'delta_y', 'object_pixels', 'background_pixels', 'brightness_status')

def parse_edges(value):
    edges = [float(v.strip()) for v in value.split(',') if v.strip()]
    if not edges or len(edges) > 20 or any(not math.isfinite(v) or v <= 0 for v in edges) or edges != sorted(set(edges)):
        raise ValueError('Enter 1–20 strictly increasing positive numbers, e.g. 8,16,32')
    return edges

def bin_labels(edges):
    fmt = lambda n: f'{n:g}'
    return [f'<{fmt(edges[0])}'] + [f'{fmt(a)}–<{fmt(b)}' for a,b in zip(edges, edges[1:])] + [f'≥{fmt(edges[-1])}']

def size_bin(d, edges=(8,16,32)):
    for n, edge in enumerate(edges):
        if d < edge: return bin_labels(edges)[n]
    return bin_labels(edges)[-1]

def normalize(p):
    return str(p).strip().replace('\\', '/').lstrip('./')

def mask_for(seg, width, height):
    if isinstance(seg, list) and seg:
        im = Image.new('1', (width, height))
        draw = ImageDraw.Draw(im)
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
                draw.polygon(list(zip(poly[::2], poly[1::2])), fill=1)
        return im
    if isinstance(seg, dict) and seg.get('counts') is not None:
        from pycocotools import mask as mask_utils
        rle = seg
        if isinstance(seg['counts'], list):
            rle = mask_utils.frPyObjects(seg, height, width)
        arr = mask_utils.decode(rle)
        if arr.ndim == 3:
            arr = np.any(arr, axis=2)
        return Image.fromarray((arr > 0).astype('uint8') * 255).convert('1')
    return None


def measure_brightness(y, mask, excluded, settings=(RING_RADIUS, MIN_OBJECT_PIXELS, MIN_BACKGROUND_PIXELS)):
    """Measure a single object against an exact Euclidean ring on the working grid."""
    rows, cols = np.nonzero(mask)
    count = len(rows)
    if not count:
        return {'object_pixels': 0, 'brightness_status': 'empty_mask'}
    r, min_object, min_background = settings
    top, bottom = max(0, rows.min()-r), min(mask.shape[0], rows.max()+r+1)
    left, right = max(0, cols.min()-r), min(mask.shape[1], cols.max()+r+1)
    target = mask[top:bottom, left:right]
    height, width = target.shape
    padded = np.pad(target, r)
    prefix = np.pad(np.cumsum(padded, axis=1, dtype=np.int32), ((0,0),(1,0)))
    ring = np.zeros_like(target)
    for dy in range(-r, r+1):
        span = math.isqrt(r*r-dy*dy)
        band = prefix[r+dy:r+dy+height]
        ring |= band[:,r+span+1:r+span+1+width] > band[:,r-span:r-span+width]
    ring &= ~(target | excluded[top:bottom, left:right])
    local_y = y[top:bottom, left:right]
    background_count = int(ring.sum())
    result = {'object_pixels': count, 'background_pixels': background_count}
    if count >= min_object:
        result['object_y'] = float(np.median(local_y[target]))
    if background_count >= min_background:
        result['background_y'] = float(np.median(local_y[ring]))
    if 'object_y' not in result:
        result['brightness_status'] = 'insufficient_object_pixels'
    elif 'background_y' not in result:
        result['brightness_status'] = 'insufficient_background_pixels'
    else:
        result['delta_y'] = result['background_y'] - result['object_y']
        result['brightness_status'] = 'measured'
    return result


class Dataset:
    def __init__(self, root, gt, splits, patterns, regex_target='name', work=(960,768), progress=None):
        self.root = Path(root).resolve(); self.work = work
        self.grouping_patterns = tuple(patterns)
        self.grouping_target = regex_target
        self.progress = progress or (lambda *args: None)
        self.progress('Discovering images', None, None)
        self.images = {}; self.objects = []; self.issues = []; self.categories = {}
        self.by_name = defaultdict(list)
        for count, p in enumerate(self.root.rglob('*'), 1):
            if count % 250 == 0: self.progress(f'Discovering files: {count:,}', None, None)
            if p.is_file() and p.suffix.lower() in EXTS:
                rel = p.relative_to(self.root).as_posix(); self.by_name[p.name].append(rel)
                self.images[rel] = {'path':p, 'gt':False, 'split':'unassigned', 'groups':{}, 'matches':[], 'annotations':[], 'gt_sources':[], 'gt_invalid':False}
        rules = [re.compile(p) for p in self.grouping_patterns if p.strip()]
        for index, (rel, item) in enumerate(self.images.items(), 1):
            if index == 1 or index % 100 == 0 or index == len(self.images): self.progress('Grouping images', index, len(self.images))
            subject = Path(rel).name if regex_target == 'name' else rel
            matches = [(n, m) for n, r in enumerate(rules, 1) if (m := r.search(subject))]
            item['matches'] = [n for n,_ in matches]
            item['has_grouping_rules'] = bool(rules)
            if matches: item['groups'] = matches[0][1].groupdict()
            elif rules: self.issues.append(('regex unmatched', rel))
            if len(matches)>1: self.issues.append(('regex multiple', f'{rel}: {item["matches"]}'))
        gtpath = Path(gt) if gt else None
        files = sorted(gtpath.rglob('*.json')) if gtpath and gtpath.is_dir() else [gtpath] if gtpath and gtpath.is_file() else []
        for index, fp in enumerate(files, 1):
            self.progress('Loading GT', index - 1, len(files))
            try: self.load_gt(fp)
            except Exception as e: self.issues.append(('invalid GT', f'{fp}: {e}'))
        self.progress('Loading split indexes', None, None)
        assignments = defaultdict(set)
        for split, filename in splits.items():
            if not filename: continue
            try:
                with open(filename, encoding='utf-8-sig') as f:
                    for line_no, row in enumerate(csv.reader(f),1):
                        if line_no % 500 == 0: self.progress(f'Loading {split} index: {line_no:,} lines', None, None)
                        if not row or not row[0].strip() or row[0].lstrip().startswith('#'): continue
                        key = self.resolve(row[0])
                        if key: assignments[key].add(split)
                        else: self.issues.append(('split missing or ambiguous', f'{split}:{line_no}: {row[0]}'))
            except (OSError, csv.Error) as e: self.issues.append(('split error', f'{filename}: {e}'))
        for rel, values in assignments.items():
            if len(values) == 1: self.images[rel]['split'] = next(iter(values))
            else: self.issues.append(('split conflict', f'{rel}: {sorted(values)}'))
        for rel,item in self.images.items():
            if not item['gt_sources']: self.issues.append(('missing GT', rel))

    def resolve(self, value):
        name = normalize(value)
        if name in self.images: return name
        candidates = self.by_name.get(Path(name).name, [])
        return candidates[0] if len(candidates)==1 else None

    def load_gt(self, fp):
        data = json.loads(fp.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or not ('images' in data or 'image' in data) or not isinstance(data.get('annotations'), list):
            if isinstance(data, dict):
                candidates = data.get('images', [data.get('image')])
                if isinstance(candidates, list):
                    for img in candidates:
                        rel = self.resolve(img.get('file_name', '')) if isinstance(img, dict) else None
                        if rel:
                            self.images[rel]['gt_sources'].append(str(fp)); self.images[rel]['gt_invalid'] = True
            raise ValueError('not a COCO annotation document')
        categories = data.get('categories', [])
        malformed_categories = not isinstance(categories, list)
        for c in categories if isinstance(categories, list) else []:
            try: self.categories[c['id']] = c.get('name', str(c['id']))
            except (KeyError, TypeError, AttributeError): malformed_categories = True
        records = data.get('images', [data.get('image')]); ann_by_id = defaultdict(list)
        if not isinstance(records, list): raise ValueError('images must be a list')
        malformed_annotations = any(not isinstance(ann, dict) for ann in data['annotations'])
        for ann in data['annotations']:
            if not isinstance(ann, dict): continue
            ann_by_id[repr(ann.get('image_id'))].append(ann)
        record_ids = Counter(repr(img.get('id')) for img in records if isinstance(img, dict))
        orphan_ids = set(ann_by_id) - record_ids.keys()
        if orphan_ids: self.issues.append(('orphan annotations', f'{fp}: image IDs {sorted(map(str, orphan_ids))}'))
        for record_index, img in enumerate(records, 1):
            if len(records) > 1 and (record_index == 1 or record_index % 20 == 0): self.progress(f'GT {fp.name}: images', record_index, len(records))
            if not isinstance(img, dict): continue
            rel = self.resolve(img.get('file_name',''))
            if rel is None:
                self.issues.append(('GT image missing or ambiguous', f'{fp}: {img.get("file_name")}')); continue
            item = self.images[rel]
            item['gt_sources'].append(str(fp))
            if orphan_ids or malformed_annotations or malformed_categories or not isinstance(img.get('id'), (str, int, float)):
                item['gt_invalid'] = True
            image_id = repr(img.get('id'))
            if record_ids[image_id] > 1:
                item['gt_invalid'] = True
                self.issues.append(('duplicate COCO image ID', f'{fp}: {rel}: {image_id}'))
            if len(item['gt_sources']) > 1:
                self.issues.append(('duplicate GT', f'{rel}: {fp}')); continue
            try:
                w, h = int(img['width']), int(img['height'])
                if w <= 0 or h <= 0: raise ValueError('invalid dimensions')
                with Image.open(item['path']) as im: actual = im.size
            except Exception as e:
                item['gt_invalid'] = True; self.issues.append(('invalid image dimensions', f'{rel}: {e}')); continue
            item.update(gt=True, gt_size=(w,h), actual_size=actual, source=str(fp), image_id=img.get('id'))
            if actual != (w,h): self.issues.append(('dimension mismatch (scaled)', f'{rel}: GT {w}x{h}, image {actual[0]}x{actual[1]}'))
            anns = ann_by_id.get(image_id, []) if record_ids[image_id] == 1 else []
            for i, ann in enumerate(anns):
                if i == 0 or i % 20 == 0: self.progress(f'GT {fp.name}: annotations', i, len(anns))
                try:
                    x,y,bw,bh = map(float, ann['bbox'])
                    if not all(math.isfinite(v) for v in (x,y,bw,bh)) or bw < 0 or bh < 0: raise ValueError('invalid bbox')
                    if not isinstance(ann.get('category_id'), (str, int)): raise ValueError('invalid category_id')
                    sx,sy=self.work[0]/w,self.work[1]/h
                    ident=hashlib.sha256(f'{rel}|{ann.get("id",i)}'.encode()).hexdigest()[:16]
                    area = None; seg=ann.get('segmentation'); warning=''
                    try:
                        m = mask_for(seg,w,h)
                        if m is not None:
                            area = int(np.count_nonzero(np.asarray(m.resize(self.work, Image.Resampling.NEAREST))))
                            if area == 0: warning='mask disappeared at working size'
                    except Exception as e: warning=f'mask error: {e}'
                    if warning: self.issues.append(('mask warning', f'{rel} annotation {ann.get("id")}: {warning}'))
                    obj={'id':ident,'image':rel,'annotation_id':ann.get('id',i),'category_id':ann.get('category_id'), 'bbox':[x,y,bw,bh], 'width':bw*sx,'height':bh*sy,'diagonal':math.hypot(bw*sx,bh*sy),'area':area,'segmentation':seg, 'gt_size':(w,h),'actual_size':actual,'source':str(fp)}
                    obj['iscrowd'] = bool(ann.get('iscrowd', 0))
                    obj['bin']=size_bin(obj['diagonal']); self.objects.append(obj); item['annotations'].append(obj)
                except (KeyError, TypeError, ValueError, OverflowError) as e:
                    item['gt_invalid'] = True; self.issues.append(('invalid annotation', f'{rel}: {e}'))

    def analyze_brightness(self, settings=(RING_RADIUS, MIN_OBJECT_PIXELS, MIN_BACKGROUND_PIXELS)):
        self.brightness_settings = settings
        items = [(rel, item) for rel, item in self.images.items() if item['annotations']]
        for index, (rel, item) in enumerate(items, 1):
            self.progress('Measuring brightness', index-1, len(items))
            objects = item['annotations']
            for obj in objects:
                for field in BRIGHTNESS_FIELDS:
                    obj.pop(field, None)
            if gt_status(item) != 'valid':
                for obj in objects: obj['brightness_status'] = 'uncertain_gt'
                continue
            try:
                with Image.open(item['path']) as source:
                    rgb = np.asarray(source.convert('RGB').resize(self.work, Image.Resampling.BILINEAR))
                y = rgb @ np.array([.2126, .7152, .0722])
            except (OSError, ValueError) as exc:
                self.issues.append(('brightness RGB unavailable', f'{rel}: {exc}'))
                for obj in objects: obj['brightness_status'] = 'rgb_unavailable'
                continue
            class_masks = {}
            masks = []
            for obj in objects:
                try:
                    mask = mask_for(obj['segmentation'], *obj['gt_size'])
                    mask = np.asarray(mask.resize(self.work, Image.Resampling.NEAREST), dtype=bool) if mask is not None else None
                except Exception:
                    mask = None
                masks.append(mask)
                if mask is not None:
                    category = obj['category_id']
                    if category not in class_masks:
                        class_masks[category] = mask.copy()
                    else:
                        class_masks[category] |= mask
            vessel_mask = np.zeros(y.shape, dtype=bool)
            for category, mask in class_masks.items():
                if str(self.categories.get(category, '')).strip().casefold() in {'boat', 'own boat'}:
                    vessel_mask |= mask
            for obj, mask in zip(objects, masks):
                if obj['iscrowd']:
                    obj['brightness_status'] = 'crowd'
                elif mask is None:
                    obj['brightness_status'] = 'mask_unavailable'
                else:
                    obj.update(measure_brightness(y, mask, vessel_mask | class_masks[obj['category_id']], settings))
        self.progress('Measuring brightness', len(items), len(items))


def alpha_name(index):
    result = ''
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def assign_scenes(images, rules, target='name'):
    """Assign first matching rule and retain all matches for review."""
    compiled = [(name.strip() or alpha_name(n), re.compile(pattern))
                for n, (name, pattern) in enumerate(rules)]
    if not compiled:
        values = sorted({i['groups'].get('scene') for i in images.values() if i['groups'].get('scene')})
        inferred = {value: alpha_name(n) for n, value in enumerate(values)}
    for rel, item in images.items():
        item['has_scene_rules'] = bool(compiled)
        if compiled:
            subject = Path(rel).name if target == 'name' else rel
            matches = [(n + 1, name) for n, (name, regex) in enumerate(compiled) if regex.search(subject)]
            item['scene'] = matches[0][1] if matches else '(unmatched)'
            item['scene_matches'] = matches
        else:
            raw = item['groups'].get('scene')
            item['scene'] = inferred.get(raw, '(unmatched)')
            item['scene_matches'] = []


SPECIAL_BIN = '8–<16 and height <4'


def object_matches_bin(obj, bin_name, edges):
    if bin_name == 'All': return True
    if bin_name == SPECIAL_BIN: return 8 <= obj['diagonal'] < 16 and obj['height'] < 4
    return size_bin(obj['diagonal'], edges) == bin_name


def object_matches_filter(obj, category, bin_name, edges, mode='Any', brightness=None):
    category_matches = category == 'All' or str(obj['category_id']) == str(category)
    if not category_matches or mode == 'Without category':
        return category_matches
    if not object_matches_bin(obj, bin_name, edges):
        return False
    if brightness is None:
        return True
    kind, lower, upper = brightness
    delta = obj.get('delta_y')
    if delta is None:
        return False
    if (kind == 'Positive' and delta <= 0) or (kind == 'Negative' and delta >= 0):
        return False
    value = abs(delta) if kind == 'Absolute' else delta
    return lower <= value <= upper


def gt_status(item):
    sources = item.get('gt_sources', [])
    if len(sources) > 1: return 'duplicate + invalid' if item.get('gt_invalid') else 'duplicate'
    if item.get('gt_invalid'): return 'invalid'
    return 'valid' if item.get('gt') else 'missing'


def match_details(item, category, bin_name, edges, mode='Any', brightness=None):
    objects = [o for o in item['annotations'] if category == 'All' or str(o['category_id']) == str(category)]
    total = len(objects)
    count = sum(object_matches_filter(o, category, bin_name, edges, mode, brightness) for o in objects)
    if mode == 'Without category':
        return gt_status(item) == 'valid' and total == 0, count, total, False
    if category == 'All' and bin_name == 'All' and mode == 'Any' and brightness is None:
        return True, count, total, False
    if not total: return False, count, total, False
    if mode == 'All': return count == total, count, total, False
    if mode == 'Mode':
        if bin_name == 'All': return count > 0, count, total, False
        buckets = Counter()
        for obj in objects:
            # Keep the special sub-bin disjoint from all ordinary bins.
            label = SPECIAL_BIN if bin_name == SPECIAL_BIN and object_matches_bin(obj, SPECIAL_BIN, edges) else size_bin(obj['diagonal'], edges)
            buckets[label] += 1
        maximum = max(buckets.values())
        qualifies = count > 0 and buckets[bin_name] == maximum
        return qualifies, count, total, qualifies and sum(v == maximum for v in buckets.values()) > 1
    if mode == 'Any': return count > 0, count, total, False
    raise ValueError(f'Unknown filter mode: {mode}')


def filter_images(ds, category, bin_name, scene, split, edges, mode='Any', brightness=None):
    selected = []
    for rel, item in ds.images.items():
        if scene != 'All' and item.get('scene', '(unmatched)') != scene: continue
        if split != 'All' and item['split'] != split: continue
        if match_details(item, category, bin_name, edges, mode, brightness)[0]: selected.append(rel)
    return selected


def objects_in_images(ds, image_paths, category):
    return [o for rel in image_paths for o in ds.images[rel]['annotations']
            if category == 'All' or str(o['category_id']) == str(category)]


def brightness_summary(objects):
    values = np.array([obj['delta_y'] for obj in objects if 'delta_y' in obj])
    counts = Counter(obj.get('brightness_status', 'not_measured') for obj in objects if 'delta_y' not in obj)
    return values, counts


def regex_status(item):
    if item.get('has_scene_rules'):
        hits = item['scene_matches']
        return ('Multiple scene rules matched' if len(hits) > 1 else
                'Scene rule matched' if hits else 'No scene rule matched')
    hits = item.get('matches', [])
    return ('Multiple grouping rules matched' if len(hits) > 1 else 'Grouping rule matched' if hits else
            'No grouping match' if item.get('has_grouping_rules') else 'No regex configured')


def export_report(ds, out, category, objects, edges=(8,16,32), copy_images=False, image_paths=None, filter_context=None):
    image_paths = list({o['image'] for o in objects} if image_paths is None else image_paths)
    out = Path(out).resolve()
    if out == ds.root or ds.root in out.parents:
        raise ValueError('Choose a report folder outside the image dataset root to protect source files.')
    out.mkdir(parents=True, exist_ok=True)
    fields=['id','image','annotation_id','category_id','source','split','groups','gt_size','actual_size','work_size','bbox','width','height','diagonal','area','bin',*BRIGHTNESS_FIELDS]
    with (out/'objects.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for o in objects:
            row={k:o.get(k) for k in fields}; row.update(split=ds.images[o['image']]['split'],groups=json.dumps({**ds.images[o['image']]['groups'], 'scene':ds.images[o['image']].get('scene','(unmatched)')},ensure_ascii=False),work_size=ds.work,bin=size_bin(o['diagonal'],edges),brightness_status=o.get('brightness_status','not_measured')); writer.writerow(row)
    with (out/'images.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['image','split','scene','regex_status','grouping_rule_hits','scene_rule_hits','gt_sources','gt_status','selected_category_object_count','measured_objects','min_delta_y','max_delta_y'])
        writer.writeheader()
        by_image = defaultdict(list)
        for obj in objects: by_image[obj['image']].append(obj)
        for rel in sorted(image_paths):
            item=ds.images[rel]
            values, _ = brightness_summary(by_image[rel])
            writer.writerow({'image':rel,'split':item['split'],'scene':item.get('scene','(unmatched)'),
                'regex_status':regex_status(item), 'grouping_rule_hits':json.dumps(item.get('matches',[])),
                'scene_rule_hits':json.dumps(item.get('scene_matches',[]),ensure_ascii=False),
                'gt_sources':json.dumps(item.get('gt_sources', []),ensure_ascii=False), 'gt_status':gt_status(item),
                'selected_category_object_count':len(by_image[rel]), 'measured_objects':len(values),
                'min_delta_y':values.min() if len(values) else '', 'max_delta_y':values.max() if len(values) else ''})
    groups=defaultdict(list)
    for o in objects: groups[(ds.images[o['image']].get('scene', '(unmatched)'),size_bin(o['diagonal'],edges))].append(o)
    with (out/'groups.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['scene','bin','objects','images','object_denominator','image_denominator'])
        for (scene,b), vals in sorted(groups.items()): w.writerow([scene,b,len(vals),len({v['image'] for v in vals}),len(objects),len(image_paths)])
    copied=[]
    if copy_images:
        for rel in sorted(image_paths):
            dest=out/'images'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.resolve() == ds.images[rel]['path'].resolve(): raise ValueError('Report image would overwrite a source image.')
            shutil.copy2(ds.images[rel]['path'], dest)
            copied.append(rel)
    has={o['image'] for o in objects}
    statuses=Counter(gt_status(ds.images[rel]) for rel in image_paths)
    valid_has=sum(rel in has for rel in image_paths if gt_status(ds.images[rel]) == 'valid')
    write_report_html(ds, out, category, objects, image_paths, edges, groups, statuses,
                      valid_has, set(copied), filter_context or {})


def write_report_html(ds, out, category, objects, image_paths, edges, groups, statuses, valid_has, copied, context):
    def esc(value):
        return html.escape(str(value))

    def table(headers, rows):
        head = ''.join(f'<th scope="col">{esc(h)}</th>' for h in headers)
        body = ''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in row)+'</tr>' for row in rows)
        return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    def image_link(rel):
        href = 'images/'+quote(rel, safe='/') if rel in copied else ds.images[rel]['path'].resolve().as_uri()
        return f'<a href="{esc(href)}">{esc(rel)}</a>'

    values, unknown = brightness_summary(objects)
    selected = set(image_paths)
    composition = defaultdict(list)
    dataset_counts = Counter((item['split'],item.get('scene','(unmatched)')) for item in ds.images.values())
    for rel in image_paths:
        item = ds.images[rel]
        composition[item['split'], item.get('scene', '(unmatched)')].append(rel)
    group_objects = defaultdict(list)
    for obj in objects:
        item = ds.images[obj['image']]
        group_objects[item['split'], item.get('scene','(unmatched)')].append(obj)
    composition_rows = []
    for (split, scene), rels in sorted(composition.items()):
        subset = group_objects[split,scene]
        measured, _ = brightness_summary(subset)
        composition_rows.append((split, scene, len(rels), dataset_counts[split,scene], f'{len(rels)/len(image_paths):.1%}',
                                 len(subset), len(measured), f'{np.median(measured):+.1f}' if len(measured) else '—'))
    size_rows = [(scene, label, len(items), len({o['image'] for o in items}),
                  f'{len(items)/len(objects):.1%}') for (scene,label),items in sorted(groups.items())]
    rgb_settings = getattr(ds, 'brightness_settings', None)
    protocol = (f'Ring radius {rgb_settings[0]} px · minimum object/background samples '
                f'{rgb_settings[1]}/{rgb_settings[2]} px · Y = 0.2126R + 0.7152G + 0.0722B (0–255 RGB)'
                if rgb_settings else 'Brightness measurement disabled at scan')
    summary = []
    if len(values):
        absolute = np.abs(values)
        summary = [('Positive (object darker)', int((values > 0).sum())),
                   ('Zero', int((values == 0).sum())), ('Negative (object brighter)', int((values < 0).sum())),
                   ('Median ΔY', f'{np.median(values):+.1f}'), ('ΔY P10 / P90', f'{np.percentile(values,10):+.1f} / {np.percentile(values,90):+.1f}'),
                   ('Median |ΔY|', f'{np.median(absolute):.1f}'),
                   ('|ΔY| P10 / P90', f'{np.percentile(absolute,10):.1f} / {np.percentile(absolute,90):.1f}')]
    summary += [('Unknown · '+key.replace('_',' '), n) for key,n in sorted(unknown.items())]
    histograms = []
    if len(values):
        for title, samples, bounds, fmt in [
            ('Signed ΔY', values, np.linspace(-255,255,21), '+.1f'),
            ('Absolute |ΔY|', np.abs(values), np.linspace(0,255,11), '.1f'),
        ]:
            bins, _ = np.histogram(samples, bins=bounds)
            peak = max(1,int(bins.max()))
            bars = ''.join(f'<li><span>{lo:{fmt}} to {hi:{fmt}}</span>'
                           f'<meter min="0" max="{peak}" value="{n}"></meter><b>{n}</b></li>'
                           for lo,hi,n in zip(bounds,bounds[1:],bins))
            histograms.append(f'<details open><summary>{title} histogram · counts of measured objects</summary><ul class="bars">{bars}</ul></details>')
    examples = sorted((obj for obj in objects if 'delta_y' in obj), key=lambda obj:abs(obj['delta_y']))[:20]
    examples_html = ''.join(f'<li>{image_link(obj["image"])} · GT {esc(obj["id"])} · ΔY {obj["delta_y"]:+.1f} · bbox {obj["diagonal"]:.1f} px</li>' for obj in examples)
    regex_rows = sorted(Counter(regex_status(ds.images[rel]) for rel in image_paths).items())
    case_rows = []
    object_counts = Counter(obj['image'] for obj in objects)
    measured_counts = Counter(obj['image'] for obj in objects if 'delta_y' in obj)
    for rel in sorted(image_paths):
        item = ds.images[rel]
        cells = (item['split'], item.get('scene','(unmatched)'), gt_status(item), regex_status(item),
                 object_counts[rel], measured_counts[rel])
        case_rows.append('<tr><td>'+image_link(rel)+'</td>'+''.join(f'<td>{esc(cell)}</td>' for cell in cells)+'</tr>')
    issues = []
    for kind, detail in ds.issues:
        if kind.startswith('regex '):
            rel = detail if kind == 'regex unmatched' else detail.rsplit(': ',1)[0]
            if rel not in selected or ds.images[rel].get('has_scene_rules'):
                continue
            kind = kind.replace('regex ', 'grouping regex ')
        issues.append((kind, detail))
    category_label = next((f'{name} ({key})' for key,name in ds.categories.items() if str(key)==str(category)),category)
    filters = table(('Setting','Value'), [('Category', category_label),
                                          *[(k,v) for k,v in context.items() if k != 'category'],
                                          ('Working grid', f'{ds.work[0]}×{ds.work[1]} px'), ('Diagonal bin edges', edges)])
    rule_rows = [(f'Grouping · {getattr(ds,"grouping_target","name")}', n, rule)
                 for n,rule in enumerate(getattr(ds,'grouping_patterns',()),1)]
    scene_target = getattr(ds,'applied_scene_target','name')
    rule_rows += [(f'Scene · {scene_target}', name or alpha_name(n), rule)
                  for n,(name,rule) in enumerate(getattr(ds,'applied_scene_rules',()))]
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seg Dataset Lab · Dataset report</title><style>
body{{font:15px/1.5 system-ui,sans-serif;color:#172b35;background:#f2f6f7;margin:0}}
main{{max-width:1080px;margin:auto;padding:24px}} section{{background:white;border:1px solid #cfdbde;border-radius:12px;padding:18px;margin:16px 0;overflow:auto}}
h1,h2{{line-height:1.2;margin-top:0}} table{{border-collapse:collapse;width:100%}} th,td{{padding:.5em .75em;text-align:left;border-bottom:1px solid #dce6e8}} th{{background:#eaf2f4}}
meter{{width:100%;accent-color:#13746a}} .bars{{list-style:none;padding:0}} .bars li{{display:grid;grid-template-columns:8em minmax(4em,30em) 3em;gap:.6em}}
a{{color:#075d70}} nav a{{margin-right:1em}} summary{{cursor:pointer}}
</style><main><header><h1>Dataset composition</h1><p><strong>{len(image_paths):,}</strong> selected images · <strong>{len(objects):,}</strong> category objects · <strong>{len(values):,}</strong> measured ΔY</p>
<nav><a href="images.csv">Images CSV</a><a href="objects.csv">Objects CSV</a><a href="groups.csv">Groups CSV</a></nav></header>
<section><h2>Selection and GT coverage</h2>{filters}<p>{len(image_paths):,} of {len(ds.images):,} dataset images. Valid GT with target: {valid_has}; valid GT without target: {statuses['valid']-valid_has}.</p>
{table(('GT status','Images'), [(key,statuses[key]) for key in ('valid','missing','invalid','duplicate','duplicate + invalid')])}
<p>Object summaries include all objects of the selected category in selected images, including objects outside the active size or ΔY range.</p></section>
<section><h2>Split × scene</h2>{table(('Split','Scene','Selected images','Dataset images in group','Share of selected images','Objects','Measured ΔY','Median ΔY'),composition_rows)}</section>
<section><h2>Scene × size</h2>{table(('Scene','Diagonal bin (px)','Objects','Images with an object in bin','Share of selected objects'),size_rows)}
<p>An image can contribute to multiple size bins.</p></section>
<section><h2>Object-to-background brightness</h2><p>ΔY = background ring median − object median. {esc(protocol)}.</p>
{table(('Measure','Value'),summary)}
{''.join(histograms) if histograms else '<p>No measured ΔY in this selection.</p>'}
{('<h3>Cases nearest zero</h3><ol>'+examples_html+'</ol>') if examples else ''}
<p>{len(copied)} images copied. Case links open local images.</p></section>
<section><h2>Regex coverage</h2>{table(('Effective match status','Selected images'),regex_rows)}
<p>Explicit scene rules determine scene coverage when configured. Images CSV retains both scene and grouping rule hits.</p>
<details><summary>Regex definitions</summary>{table(('Rule scope','Rule','Pattern'),rule_rows)}</details></section>
<section><details><summary>Selected images ({len(image_paths):,})</summary>
<p><label>Search images <input id="image-search" type="search" placeholder="Filename, split, scene or GT status"></label> <output id="image-count">{len(image_paths):,}</output></p>
<table><thead><tr><th>Image</th><th>Split</th><th>Scene</th><th>GT</th><th>Regex status</th><th>Objects</th><th>Measured ΔY</th></tr></thead>
<tbody id="image-index">{''.join(case_rows)}</tbody></table></details></section>
<section><details><summary>Diagnostics ({len(issues):,})</summary>{table(('Type','Detail'),issues)}</details></section></main>
<script>
const search = document.getElementById('image-search');
const rows = Array.from(document.getElementById('image-index').rows, row => [row, row.textContent.toLowerCase()]);
search.addEventListener('input', () => {{
  const terms = search.value.trim().toLowerCase().split(/\\s+/);
  let visible = 0;
  for (const [row, content] of rows) {{
    row.hidden = !terms.every(term => content.includes(term));
    if (!row.hidden) visible++;
  }}
  document.getElementById('image-count').value = visible.toLocaleString();
}});
</script></html>"""
    (out/'report.html').write_text(report,encoding='utf-8')

def object_sort_key(obj, column, categories):
    if column == 'id': return str(obj['id']).casefold()
    if column == 'class': return str(categories.get(obj['category_id'], obj['category_id'])).casefold()
    if column == 'size': return obj['diagonal']
    if column == 'area': return obj['area'] if obj['area'] is not None else -1
    if column == 'delta_y': return obj.get('delta_y', -math.inf)
    raise ValueError(f'Unknown object column: {column}')


def preview_to_work(x, y, canvas_size, display_size, offset, work_size):
    cw,ch=canvas_size; dw,dh=display_size
    left=cw/2+offset[0]-dw/2; top=ch/2+offset[1]-dh/2
    if not (left <= x < left+dw and top <= y < top+dh): return None
    return min(work_size[0]-1,int((x-left)*work_size[0]/dw)), min(work_size[1]-1,int((y-top)*work_size[1]/dh))


def hit_mask(objects, masks, x, y):
    hits=[(int(mask.sum()), index, obj) for index,obj in enumerate(objects)
          if (mask := masks.get(id(obj))) is not None and mask[y,x]]
    return min(hits, key=lambda entry:(entry[0],entry[1]))[2] if hits else None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Seg Dataset Lab'); self.geometry('1500x900')
        self.ds=None; self.rows=[]; self.image_rows=[]; self.object_rows=[]
        self.current_image=None; self.current=None; self.photo=None
        self.zoom=1.; self.offset=[0,0]; self.drag=None; self.events=queue.Queue(); self.scanning=False
        self.edges=[8.,16.,32.]; self.chart=None; self.chart_bars=[]
        self.scene_rules=[]; self.sort_column='image'; self.sort_desc=False
        self.gt_sort_column='id'; self.gt_sort_desc=False; self.mask_cache={}; self.preview_size=None; self.drag_start=None; self.dragged=False
        self.vars={k:tk.StringVar(value=v) for k,v in {'root':'','gt':'','train':'','val':'','test':'','regex':r'^(?P<scene>.+?)_(?P<camera>camera\d+)_frame_(?P<frame>\d+)','target':'name','width':'960','height':'768','category':'1','bin':'All','scene':'All','split':'All','lookup':'','edges':'8,16,32','mode':'Any'}.items()}
        top=ttk.Frame(self,padding=5); top.pack(fill='x')
        for r,(key,label,isdir) in enumerate([('root','Images folder',True),('gt','GT JSON / folder',False),('train','Train index',False),('val','Val index',False),('test','Test index',False)]):
            ttk.Label(top,text=label).grid(row=r,column=0,sticky='w'); ttk.Entry(top,textvariable=self.vars[key]).grid(row=r,column=1,sticky='ew'); ttk.Button(top,text='Browse',command=lambda k=key,d=isdir:self.choose(k,d)).grid(row=r,column=2)
            if key=='gt': ttk.Button(top,text='Folder',command=lambda:self.choose('gt',True)).grid(row=r,column=3)
        top.columnconfigure(1,weight=1)
        rules=ttk.Frame(self,padding=5); rules.pack(fill='x')
        ttk.Label(rules,text='Regex (;; separates rules)').pack(side='left'); ttk.Entry(rules,textvariable=self.vars['regex'],width=25).pack(side='left',fill='x',expand=True)
        ttk.Combobox(rules,textvariable=self.vars['target'],values=['name','path'],width=6,state='readonly').pack(side='left')
        for k in ('width','height'): ttk.Entry(rules,textvariable=self.vars[k],width=5).pack(side='left')
        self.scan_button=ttk.Button(rules,text='Scan',command=self.scan); self.scan_button.pack(side='left')
        for title,cmd in [('Save project',self.save),('Open project',self.open),('Scenes',self.scene_editor),('Statistics',self.show_chart),('Export report',self.export)]: ttk.Button(rules,text=title,command=cmd).pack(side='left')
        filters=ttk.Frame(self,padding=5); filters.pack(fill='x'); self.boxes={}
        for k,vals in [('category',['All']),('bin',['All',*BINS,'8–<16 and height <4']),('mode',['Any','All','Mode','Without category']),('scene',['All']),('split',['All','train','val','test','unassigned'])]:
            ttk.Label(filters,text=k.title()).pack(side='left'); c=ttk.Combobox(filters,textvariable=self.vars[k],values=vals,width=13,state='readonly'); c.pack(side='left'); c.bind('<<ComboboxSelected>>',lambda e:self.refresh()); self.boxes[k]=c
        ttk.Label(filters,text='Object ID').pack(side='left'); ttk.Entry(filters,textvariable=self.vars['lookup'],width=17).pack(side='left'); ttk.Button(filters,text='Find',command=self.find).pack(side='left')
        self.measure_brightness = tk.BooleanVar(value=False)
        brightness_bar = ttk.Frame(self,padding=5); brightness_bar.pack(fill='x')
        ttk.Checkbutton(brightness_bar,text='Measure brightness on scan',variable=self.measure_brightness).pack(side='left')
        self.brightness_filter = None
        for key, label, default in [('radius','Ring (px)','12'),('min_object','Min object px','32'),('min_background','Min background px','64')]:
            self.vars[key] = tk.StringVar(value=default)
            ttk.Label(brightness_bar,text=label).pack(side='left',padx=(10,2))
            ttk.Entry(brightness_bar,textvariable=self.vars[key],width=5).pack(side='left')
        self.vars['delta_kind'] = tk.StringVar(value='Off')
        self.vars['delta_min'] = tk.StringVar()
        self.vars['delta_max'] = tk.StringVar()
        ttk.Label(brightness_bar,text='ΔY filter').pack(side='left',padx=(12,2))
        self.delta_box = ttk.Combobox(brightness_bar,textvariable=self.vars['delta_kind'],
                                     values=['Off','Signed','Positive','Negative','Absolute'],state='disabled',width=9)
        self.delta_box.pack(side='left')
        for key, label in [('delta_min','Min'),('delta_max','Max')]:
            ttk.Label(brightness_bar,text=label).pack(side='left',padx=(4,0))
            ttk.Entry(brightness_bar,textvariable=self.vars[key],width=6).pack(side='left',padx=2)
        ttk.Button(brightness_bar,text='Apply ΔY range',command=self.apply_brightness_filter).pack(side='left')
        body=tk.PanedWindow(self,orient='horizontal',sashwidth=9,sashrelief='raised',showhandle=True); body.pack(fill='both',expand=True)
        images=ttk.Frame(body); objects=ttk.Frame(body); right=ttk.Frame(body)
        body.add(images,minsize=100,width=420); body.add(objects,minsize=100,width=340); body.add(right,minsize=160)
        ttk.Label(images,text='Images (select to view)',font=('TkDefaultFont',10,'bold')).pack(anchor='w')
        self.image_tree=self.make_tree(images,[('image',200),('scene',75),('split',75),('GT',65),('match',75)])
        self.image_tree.bind('<<TreeviewSelect>>',self.select_image)
        for col in ('image','scene','split','GT','match'): self.image_tree.heading(col,command=lambda c=col:self.sort_images(c))
        ttk.Label(objects,text='GT objects in selected image',font=('TkDefaultFont',10,'bold')).pack(anchor='w')
        self.object_tree=self.make_tree(objects,[('id',105),('class',65),('size',125),('area',60),('delta_y',65)])
        self.object_tree.heading('delta_y',text='ΔY')
        self.object_tree.bind('<<TreeviewSelect>>',self.select_object)
        for col in ('id','class','size','area','delta_y'): self.object_tree.heading(col,command=lambda c=col:self.sort_objects(c))
        opts=ttk.Frame(right); opts.pack(fill='x'); self.showbox=tk.BooleanVar(value=True); self.showmask=tk.BooleanVar(value=True); self.showall=tk.BooleanVar(value=True)
        for label,var in [('BBox',self.showbox),('GT overlay',self.showmask),('All masks',self.showall)]: ttk.Checkbutton(opts,text=label,variable=var,command=self.render).pack(side='left')
        ttk.Label(opts,text='Click mask to select; wheel: zoom; drag: pan').pack(side='left')
        self.canvas=tk.Canvas(right,bg='#252525'); self.canvas.pack(fill='both',expand=True); self.canvas.bind('<Configure>',lambda e:self.render()); self.canvas.bind('<MouseWheel>',self.wheel); self.canvas.bind('<Button-4>',lambda e:self.wheel(e,1)); self.canvas.bind('<Button-5>',lambda e:self.wheel(e,-1)); self.canvas.bind('<ButtonPress-1>',self.press); self.canvas.bind('<B1-Motion>',self.pan); self.canvas.bind('<ButtonRelease-1>',self.release)
        bottom=ttk.Frame(self); bottom.pack(fill='x')
        self.status=tk.StringVar(value='Select dataset and GT, then Scan.'); ttk.Label(bottom,textvariable=self.status,anchor='w').pack(side='left',fill='x',expand=True)
        self.progress=ttk.Progressbar(bottom,length=220,mode='determinate'); self.progress.pack(side='right',padx=5)
        self.after(80,self.poll_events)
    def make_tree(self,parent,columns):
        holder=ttk.Frame(parent); holder.pack(fill='both',expand=True)
        tree=ttk.Treeview(holder,columns=[x[0] for x in columns],show='headings',selectmode='browse')
        for col,width in columns: tree.heading(col,text=col); tree.column(col,width=width,minwidth=45,anchor='w')
        vs=ttk.Scrollbar(holder,orient='vertical',command=tree.yview); hs=ttk.Scrollbar(holder,orient='horizontal',command=tree.xview); tree.configure(yscrollcommand=vs.set,xscrollcommand=hs.set)
        tree.grid(row=0,column=0,sticky='nsew'); vs.grid(row=0,column=1,sticky='ns'); hs.grid(row=1,column=0,sticky='ew'); holder.rowconfigure(0,weight=1); holder.columnconfigure(0,weight=1)
        return tree
    def choose(self,key,directory):
        value=filedialog.askdirectory() if directory else filedialog.askopenfilename()
        if value:self.vars[key].set(value)
    def scan(self):
        if self.scanning:return
        try:
            root=self.vars['root'].get(); gt=self.vars['gt'].get(); work=(int(self.vars['width'].get()),int(self.vars['height'].get()))
            if not root or not Path(root).is_dir() or min(work)<=0: raise ValueError('Choose a valid folder and positive working size')
            if gt and not Path(gt).exists(): raise ValueError('GT path does not exist')
            patterns=[s.strip() for s in self.vars['regex'].get().split(';;') if s.strip()]
            for p in patterns: re.compile(p)
            measure = self.measure_brightness.get()
            settings = tuple(int(self.vars[k].get()) for k in ('radius','min_object','min_background')) if measure else None
            if settings and (not 1 <= settings[0] <= 128 or min(settings[1:]) < 1):
                raise ValueError('Ring radius must be 1–128 px and minimum sample counts must be positive.')
        except Exception as e: messagebox.showerror('Invalid settings',str(e)); return
        splits={k:self.vars[k].get() for k in ('train','val','test')}; target=self.vars['target'].get()
        self.scanning=True; self.scan_button.state(['disabled']); self.progress.configure(mode='indeterminate'); self.progress.start(12); self.status.set('Scanning: discovering images…')
        def worker():
            last=[0.]
            def progress(phase, current, total):
                now=time.monotonic()
                if now-last[0] > .1 or current == total:
                    self.events.put(('progress',(phase,current,total))); last[0]=now
            try:
                ds = Dataset(root,gt,splits,patterns,target,work,progress)
                if settings: ds.analyze_brightness(settings)
                self.events.put(('done',ds))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()
    def poll_events(self):
        try:
            while True:
                kind,payload=self.events.get_nowait()
                if kind=='progress':
                    phase,n,total=payload
                    if total:
                        if str(self.progress['mode']) != 'determinate': self.progress.stop(); self.progress.configure(mode='determinate')
                        self.progress['value']=100*n/total; self.status.set(f'{phase}: {n:,} / {total:,} ({100*n/total:.0f}%)')
                    else:
                        if str(self.progress['mode']) != 'indeterminate': self.progress.configure(mode='indeterminate'); self.progress.start(12)
                        self.status.set(phase+'…')
                else:
                    self.progress.stop(); self.progress.configure(mode='determinate',value=100 if kind=='done' else 0); self.scanning=False; self.scan_button.state(['!disabled'])
                    if kind=='done':self.loaded(payload)
                    else:self.status.set('Scan failed'); messagebox.showerror('Scan failed',payload)
        except queue.Empty:pass
        self.after(80,self.poll_events)
    def loaded(self,ds):
        self.ds=ds; self.current=None; self.current_image=None; self.mask_cache={}; self.canvas.delete('all')
        self.boxes['category']['values']=['All']+[str(x) for x in sorted(ds.categories)]
        self.delta_box.configure(state='readonly' if hasattr(ds,'brightness_settings') else 'disabled')
        if not hasattr(ds,'brightness_settings'): self.vars['delta_kind'].set('Off')
        self.apply_brightness_filter(refresh=False)
        self.apply_scene_rules()

    def apply_brightness_filter(self, refresh=True):
        kind = self.vars['delta_kind'].get()
        try:
            brightness = None
            if kind != 'Off':
                if not self.ds or not hasattr(self.ds,'brightness_settings'):
                    raise ValueError('Scan with brightness measurement enabled first.')
                default_min = 0 if kind in {'Positive','Absolute'} else -255
                default_max = 0 if kind == 'Negative' else 255
                lower = float(self.vars['delta_min'].get().strip() or default_min)
                upper = float(self.vars['delta_max'].get().strip() or default_max)
                if not default_min <= lower <= upper <= default_max:
                    raise ValueError(f'Enter an inclusive range within {default_min} to {default_max}.')
                brightness = kind, lower, upper
        except ValueError as exc:
            if not refresh: self.brightness_filter = None
            messagebox.showerror('ΔY filter',str(exc))
            return
        self.brightness_filter = brightness
        if refresh: self.refresh()

    def apply_scene_rules(self):
        self.ds.applied_scene_rules = tuple(self.scene_rules)
        self.ds.applied_scene_target = self.vars['target'].get()
        assign_scenes(self.ds.images,self.ds.applied_scene_rules,self.ds.applied_scene_target)
        scenes=sorted({i['scene'] for i in self.ds.images.values()})
        self.boxes['scene']['values']=['All']+scenes
        if self.vars['scene'].get() not in ['All']+scenes:self.vars['scene'].set('All')
        self.refresh()

    def filtered(self, ignore_bin=False):
        if not self.ds:return []
        return objects_in_images(self.ds,self.image_rows,self.vars['category'].get())

    def sort_images(self, column):
        if self.sort_column == column:self.sort_desc=not self.sort_desc
        else:self.sort_column=column; self.sort_desc=False
        self.refresh()

    def refresh(self):
        if not self.ds:return
        category, bin_name, scene, split, mode = (self.vars[key].get() for key in ('category','bin','scene','split','mode'))
        self.image_rows=filter_images(self.ds,category,bin_name,scene,split,self.edges,mode,self.brightness_filter)
        matches = {rel:match_details(self.ds.images[rel],category,bin_name,self.edges,mode,self.brightness_filter)
                   for rel in self.image_rows}
        def sort_key(rel):
            item=self.ds.images[rel]
            details=matches[rel]
            keys={'image':rel.casefold(),'scene':item['scene'].casefold(),'split':item['split'].casefold(),'GT':len(item['annotations']) if item['gt'] else -1,'match':(details[1],details[2]) if item['gt'] else (-1,-1)}
            return keys[self.sort_column]
        self.image_rows.sort(key=sort_key,reverse=self.sort_desc)
        self.rows=self.filtered()
        self.image_tree.delete(*self.image_tree.get_children()); self.object_tree.delete(*self.object_tree.get_children()); self.object_rows=[]
        for n,rel in enumerate(self.image_rows):
            i=self.ds.images[rel]; state=gt_status(i) + (f' ({len(i["annotations"])})' if i['gt'] else '')
            _,matched,total,tied=matches[rel]
            match_text=f'{matched}/{total}'+(' tie' if tied else '')+(' ?' if gt_status(i) != 'valid' else '') if i['gt'] else '—'
            self.image_tree.insert('','end',iid=str(n),values=(rel,i['scene'],i['split'],state,match_text))
        if self.current_image in self.image_rows:
            n=self.image_rows.index(self.current_image); self.image_tree.selection_set(str(n)); self.image_tree.see(str(n)); self.populate_objects()
        else:self.current_image=None; self.current=None; self.canvas.delete('all')
        self.status.set(f'{len(self.image_rows):,} / {len(self.ds.images):,} images selected | {len(self.rows):,} category objects in selected images | {len(self.ds.issues):,} issues')
        self.draw_chart()
    def scene_editor(self):
        if not self.ds:messagebox.showinfo('Scenes','Scan a dataset first.'); return
        dialog=tk.Toplevel(self); dialog.title('Scene rules'); dialog.geometry('980x650')
        ttk.Label(dialog,text='One rule per line: Scene name | Python regex. Leave the name empty for A, B, C…; first match wins.',padding=6).pack(anchor='w')
        ttk.Label(dialog,text='Rules use the name/path selector in the main window. With no rules, existing parsed scenes receive A, B, C…; others are unmatched.').pack(anchor='w',padx=6)
        editor=tk.Text(dialog,height=6,wrap='none'); editor.pack(fill='x',padx=6,pady=5)
        editor.insert('1.0','\n'.join(f'{name} | {pattern}' for name,pattern in self.scene_rules))
        controls=ttk.Frame(dialog); controls.pack(fill='x',padx=6)
        info=tk.StringVar(value=''); ttk.Label(dialog,textvariable=info).pack(anchor='w',padx=6)
        tree=self.make_tree(dialog,[('image',310),('scene',100),('status',130),('rule hits',200)])
        def parse():
            result=[]
            for line in editor.get('1.0','end-1c').splitlines():
                if not line.strip():continue
                if '|' not in line:raise ValueError('Each line needs: name | regex. Use an empty name for automatic naming.')
                name,pattern=line.split('|',1); name=name.strip(); pattern=pattern.strip()
                if not pattern:raise ValueError('Regex cannot be empty')
                if name in ('All','(unmatched)'):raise ValueError('All and (unmatched) are reserved names')
                re.compile(pattern); result.append((name,pattern))
            return result
        def preview(commit=False):
            try:
                rules=parse()
                scratch={rel:{'groups':item['groups']} for rel,item in self.ds.images.items()}
                assign_scenes(scratch,rules,self.vars['target'].get())
                tree.delete(*tree.get_children()); counts=Counter(i['scene'] for i in scratch.values()); multi=0
                for rel,item in sorted(scratch.items()):
                    hits=item['scene_matches']; multi+=len(hits)>1
                    state='unmatched' if item['scene']=='(unmatched)' else 'multiple' if len(hits)>1 else 'matched'
                    tree.insert('','end',values=(rel,item['scene'],state,', '.join(f'{n}:{name}' for n,name in hits)))
                info.set(' | '.join(f'{scene}: {count}' for scene,count in sorted(counts.items()))+f' | Multiple matches: {multi}')
                if commit:self.scene_rules=rules; self.apply_scene_rules(); dialog.destroy()
            except (ValueError,re.error) as e:messagebox.showerror('Invalid scene rules',str(e),parent=dialog)
        ttk.Button(controls,text='Preview',command=preview).pack(side='left')
        ttk.Button(controls,text='Apply',command=lambda:preview(True)).pack(side='left')
        preview()

    def select_image(self,event=None):
        selected=self.image_tree.selection()
        if not selected:return
        rel=self.image_rows[int(selected[0])]
        if rel==self.current_image: return
        self.current_image=rel; self.current=None; self.mask_cache={}; self.zoom=1.; self.offset=[0,0]; self.populate_objects(); self.render()
    def sort_objects(self, column):
        if self.gt_sort_column == column:self.gt_sort_desc=not self.gt_sort_desc
        else:self.gt_sort_column=column; self.gt_sort_desc=False
        self.populate_objects()

    def populate_objects(self):
        self.object_tree.delete(*self.object_tree.get_children())
        if not self.current_image:return
        self.object_rows=sorted(self.ds.images[self.current_image]['annotations'],key=lambda o:object_sort_key(o,self.gt_sort_column,self.ds.categories),reverse=self.gt_sort_desc)
        for n,o in enumerate(self.object_rows):
            cls=self.ds.categories.get(o['category_id'],str(o['category_id'])); tag='match' if object_matches_filter(o,self.vars['category'].get(),self.vars['bin'].get(),self.edges,self.vars['mode'].get(),getattr(self,'brightness_filter',None)) else 'other'
            delta = f'{o["delta_y"]:+.1f}' if 'delta_y' in o else '—'
            self.object_tree.insert('','end',iid=str(n),values=(o['id'],cls,f'{o["width"]:.1f}×{o["height"]:.1f} / {o["diagonal"]:.1f}',o['area'] if o['area'] is not None else '—',delta),tags=(tag,))
        self.object_tree.tag_configure('other',foreground='#888888')
        if self.current is not None:
            for n,obj in enumerate(self.object_rows):
                if obj is self.current:self.object_tree.selection_set(str(n)); self.object_tree.see(str(n)); break
    def select_object(self,event=None):
        selected=self.object_tree.selection()
        if not selected:return
        self.current=self.object_rows[int(selected[0])]; self.render()
    def find(self):
        for o in self.rows:
            if o['id']==self.vars['lookup'].get().strip():
                n=self.image_rows.index(o['image']); self.image_tree.selection_set(str(n)); self.image_tree.see(str(n)); self.select_image()
                m=next(i for i,x in enumerate(self.object_rows) if x['id']==o['id']); self.object_tree.selection_set(str(m)); self.object_tree.see(str(m)); self.select_object(); return
        messagebox.showinfo('Find','Object ID not found in current filters.')
    def work_mask(self, obj):
        key=id(obj)
        if key not in self.mask_cache:
            try:
                mask=mask_for(obj['segmentation'],*obj['gt_size'])
                self.mask_cache[key]=np.asarray(mask.resize(self.ds.work,Image.Resampling.NEAREST),dtype=bool) if mask is not None else None
            except Exception as exc:
                self.mask_cache[key]=None
                self.status.set(f'Mask unavailable for {obj["id"]}: {exc}')
        return self.mask_cache[key]

    def render(self):
        if not self.ds or not self.current_image:return
        o=self.current; item=self.ds.images[self.current_image]
        try:
            with Image.open(item['path']) as source: im=source.convert('RGB').resize(self.ds.work,Image.Resampling.BILINEAR)
            if self.showmask.get():
                objects=item['annotations'] if self.showall.get() else ([o] if o else [])
                entries=[(obj,self.work_mask(obj)) for obj in objects]
                entries=[(obj,mask) for obj,mask in entries if mask is not None and mask.any()]
                entries.sort(key=lambda entry:(entry[0] is o, -int(entry[1].sum())))
                for obj,mask in entries:
                    digest=hashlib.sha256(str(obj['category_id']).encode()).digest()
                    color=(255,220,20) if obj is o else tuple(70+v%165 for v in digest[:3])
                    overlay=Image.new('RGB',self.ds.work,color)
                    im=Image.composite(Image.blend(im,overlay,.42),im,Image.fromarray(mask))
            if o and self.showbox.get():
                x,y,w,h=o['bbox']; sx=self.ds.work[0]/o['gt_size'][0]; sy=self.ds.work[1]/o['gt_size'][1]
                ImageDraw.Draw(im).rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='yellow',width=1)
            cw=max(self.canvas.winfo_width(),1); ch=max(self.canvas.winfo_height(),1); scale=min(cw/im.width,ch/im.height)*self.zoom
            self.preview_size=(max(1,int(im.width*scale)),max(1,int(im.height*scale)))
            im=im.resize(self.preview_size,Image.Resampling.NEAREST if scale>=1 else Image.Resampling.BILINEAR)
            self.photo=ImageTk.PhotoImage(im); self.canvas.delete('all'); self.canvas.create_image(cw/2+self.offset[0],ch/2+self.offset[1],image=self.photo)
            label=f'ID {o["id"]}  GT {o["gt_size"]} → work {self.ds.work}  mask area {o["area"]}' if o else f'{self.current_image} — click a mask or select a GT row'
            if o and 'brightness_status' in o:
                if 'delta_y' in o:
                    label += f'\nΔY {o["delta_y"]:+.1f}  Object {o["object_y"]:.1f} ({o["object_pixels"]} px)  Background {o["background_y"]:.1f} ({o["background_pixels"]} px)'
                else:
                    label += f'\nΔY unavailable: {o["brightness_status"].replace("_", " ")}'
            self.canvas.create_text(8,8,anchor='nw',fill='white',text=label)
        except Exception as e:self.status.set(f'Preview error: {e}')
    def wheel(self,e,direction=None):
        self.zoom=max(.2,min(30,self.zoom*(1.25 if (direction or e.delta)>0 else .8))); self.render()
    def press(self,e):self.drag=(e.x,e.y); self.drag_start=(e.x,e.y); self.dragged=False
    def pan(self,e):
        if self.drag:
            if self.drag_start and math.hypot(e.x-self.drag_start[0],e.y-self.drag_start[1])>3:self.dragged=True
            if self.dragged:self.offset[0]+=e.x-self.drag[0]; self.offset[1]+=e.y-self.drag[1]; self.render()
            self.drag=(e.x,e.y)
    def release(self,e):
        if self.dragged or not self.current_image or not self.showmask.get() or not self.preview_size:
            self.drag=None; return
        point=preview_to_work(e.x,e.y,(self.canvas.winfo_width(),self.canvas.winfo_height()),self.preview_size,self.offset,self.ds.work)
        if point is not None:
            objects=self.ds.images[self.current_image]['annotations'] if self.showall.get() else ([self.current] if self.current else [])
            masks={id(obj):self.work_mask(obj) for obj in objects}
            hit=hit_mask(objects,masks,*point)
            if hit is not None:
                index=next(n for n,obj in enumerate(self.object_rows) if obj is hit)
                self.object_tree.selection_set(str(index)); self.object_tree.see(str(index)); self.select_object()
        self.drag=None
    def apply_edges(self):
        try:edges=parse_edges(self.vars['edges'].get())
        except ValueError as e:messagebox.showerror('Invalid bins',str(e),parent=self.chart or self); return
        self.edges=edges; self.vars['bin'].set('All'); self.boxes['bin']['values']=['All',*bin_labels(edges),'8–<16 and height <4']; self.refresh()
    def show_chart(self):
        if not self.ds:messagebox.showinfo('Statistics','Scan a dataset first.'); return
        if self.chart and self.chart.winfo_exists():self.chart.lift(); return
        self.chart=tk.Toplevel(self); self.chart.title('Dataset statistics'); self.chart.geometry('950x660')
        controls=ttk.Frame(self.chart,padding=6); controls.pack(fill='x')
        ttk.Label(controls,text='Diagonal bin edges (px, comma-separated)').pack(side='left'); ttk.Entry(controls,textvariable=self.vars['edges'],width=25).pack(side='left'); ttk.Button(controls,text='Apply bins',command=self.apply_edges).pack(side='left')
        self.chart_note=ttk.Label(self.chart,wraplength=880,justify='left'); self.chart_note.pack(anchor='w',padx=6)
        self.stats_figure = None
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            pass
        else:
            self.stats_figure = Figure(figsize=(9,5),layout='constrained')
            self.stats_canvas = FigureCanvasTkAgg(self.stats_figure,master=self.chart)
            self.stats_canvas.get_tk_widget().pack(fill='both',expand=True)
            self.stats_canvas.mpl_connect('button_press_event',self.stats_chart_click)
            self.draw_chart()
            return
        self.chart_canvas=tk.Canvas(self.chart,bg='white'); self.chart_canvas.pack(fill='both',expand=True,padx=8,pady=8)
        self.chart_canvas.bind('<Configure>',lambda e:self.draw_chart()); self.chart_canvas.bind('<Button-1>',self.chart_click)
        self.draw_chart()
    def draw_chart(self):
        if not self.ds or not self.chart or not self.chart.winfo_exists():return
        objs=self.filtered(); labels=bin_labels(self.edges); counts=Counter(size_bin(o['diagonal'],self.edges) for o in objs)
        values, unknown = brightness_summary(objs)
        detail = f'Measured ΔY {len(values):,} / {len(objs):,} category objects'
        if len(values):
            detail += f' · median {np.median(values):+.1f} · + {int((values>0).sum()):,} / zero {int((values==0).sum()):,} / − {int((values<0).sum()):,}'
            absolute = np.abs(values)
            p10, p90 = np.percentile(absolute,[10,90])
            detail += f'\n|ΔY| median {np.median(absolute):.1f} · P10 / P90 {p10:.1f} / {p90:.1f}'
        if unknown:
            detail += '\nUnknown: ' + ', '.join(f'{key.replace("_"," ")} {count:,}' for key,count in sorted(unknown.items()))
        self.chart_note.configure(text=detail+'\nCounts include all category objects in the selected images. Click a size bar to filter.')
        if self.stats_figure is not None:
            figure = self.stats_figure
            figure.clear()
            self.size_axis = figure.add_subplot(221 if len(values) else 111)
            self.size_axis.bar(labels,[counts[label] for label in labels],color='#297b9a')
            self.size_axis.set(xlabel='BBox diagonal (px)',ylabel='Objects')
            self.size_axis.tick_params(axis='x',labelrotation=30 if len(labels)>8 else 0)
            if len(values):
                histogram = figure.add_subplot(222)
                magnitude = max(1, float(np.abs(values).max()))
                histogram.hist(values,bins=np.linspace(-magnitude,magnitude,21),color='#238174')
                histogram.axvline(0,color='#465962',linewidth=1)
                histogram.set(xlabel='Signed ΔY (background − object)',ylabel='Objects')
                abs_histogram = figure.add_subplot(223)
                abs_histogram.hist(absolute,bins=np.linspace(0,magnitude,11),color='#238174')
                abs_histogram.set(xlabel='Absolute |ΔY|',ylabel='Objects')
                scatter = figure.add_subplot(224)
                scatter.scatter([o['diagonal'] for o in objs if 'delta_y' in o],values,s=10,alpha=.35,color='#297b9a')
                scatter.axhline(0,color='#465962',linewidth=1)
                scatter.set(xlabel='BBox diagonal (px)',ylabel='Signed ΔY')
            self.stats_canvas.draw_idle()
            return
        c=self.chart_canvas; c.delete('all'); self.chart_bars=[]
        w=max(c.winfo_width(),300); h=max(c.winfo_height(),240); left=55; bottom=(h//2 if len(values) else h)-45; top=24; plotw=w-left-20; maxcount=max([counts[x] for x in labels]+[1]); step=plotw/len(labels)
        c.create_line(left,top,left,bottom,fill='#555'); c.create_line(left,bottom,w-20,bottom,fill='#555'); c.create_text(8,top,anchor='nw',text=str(maxcount))
        for n,label in enumerate(labels):
            x1=left+n*step+step*.12; x2=left+(n+1)*step-step*.12; barheight=(bottom-top-18)*counts[label]/maxcount; y=bottom-barheight
            color='#225eaa' if self.vars['bin'].get()==label else '#4e9acb'; c.create_rectangle(x1,y,x2,bottom,fill=color,outline='')
            c.create_text((x1+x2)/2,max(top,y-14),text=str(counts[label])); c.create_text((x1+x2)/2,bottom+15,text=label)
            self.chart_bars.append((x1,x2,label))
        c.create_text(w/2,bottom+35,text=f'BBox diagonal (px) · {len(self.image_rows):,} selected images',fill='#444')
        if len(values):
            magnitude = max(1,float(np.abs(values).max()))
            hist_top, hist_bottom = bottom+55, h-40
            width = (plotw-32)/2
            for x0, samples, lower, title in [(left,values,-magnitude,'Signed ΔY'),
                                              (left+width+32,absolute,0,'Absolute |ΔY|')]:
                histogram, _ = np.histogram(samples,bins=np.linspace(lower,magnitude,21))
                step = width/len(histogram)
                for index, count in enumerate(histogram):
                    x = x0+index*step
                    height = (hist_bottom-hist_top)*count/max(histogram)
                    c.create_rectangle(x,hist_bottom-height,x+step-1,hist_bottom,fill='#238174',outline='')
                c.create_line(x0,hist_bottom,x0+width,hist_bottom,fill='#555')
                c.create_text(x0,hist_bottom+12,text=f'{lower:.1f}')
                c.create_text(x0+width,hist_bottom+12,text=f'{magnitude:.1f}')
                c.create_text(x0+width/2,h-8,text=title,fill='#444')
    def chart_click(self,event):
        if not 18 <= event.y <= (self.chart_canvas.winfo_height()//2 if any('delta_y' in o for o in self.rows) else self.chart_canvas.winfo_height())-30:
            return
        for x1,x2,label in self.chart_bars:
            if x1<=event.x<=x2:self.vars['bin'].set(label); self.refresh(); return
    def stats_chart_click(self,event):
        if event.inaxes is self.size_axis and event.xdata is not None:
            labels = bin_labels(self.edges)
            index = round(event.xdata)
            if 0 <= index < len(labels):
                self.vars['bin'].set(labels[index])
                self.refresh()
    def save(self):
        p=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
        if p:
            settings={k:v.get() for k,v in self.vars.items()}; settings['edges']=','.join(f'{x:g}' for x in self.edges)
            settings['delta_kind'] = self.brightness_filter[0] if self.brightness_filter else 'Off'
            settings['delta_min'] = str(self.brightness_filter[1]) if self.brightness_filter else ''
            settings['delta_max'] = str(self.brightness_filter[2]) if self.brightness_filter else ''
            Path(p).write_text(json.dumps({'version':1,'settings':settings,'scene_rules':self.scene_rules,'measure_brightness':self.measure_brightness.get()},indent=2,ensure_ascii=False),encoding='utf-8')
    def open(self):
        p=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not p:return
        try:
            data=json.loads(Path(p).read_text(encoding='utf-8'))
            if data.get('version')!=1:raise ValueError('Unsupported project version')
            self.vars['mode'].set('Any')
            for key, default in [('radius','12'),('min_object','32'),('min_background','64'),('delta_kind','Off'),('delta_min',''),('delta_max','')]:
                self.vars[key].set(default)
            self.measure_brightness.set(data.get('measure_brightness', False))
            for k,v in data['settings'].items():
                if k in self.vars:self.vars[k].set(v)
            self.scene_rules=data.get('scene_rules',[]); self.edges=parse_edges(self.vars['edges'].get()); self.boxes['bin']['values']=['All',*bin_labels(self.edges),'8–<16 and height <4']; self.scan()
        except Exception as e:messagebox.showerror('Project error',str(e))
    def export(self):
        if not self.ds:messagebox.showinfo('Export','Scan a dataset first.'); return
        dialog=tk.Toplevel(self); dialog.title('Export report'); dialog.transient(self); dialog.grab_set()
        ttk.Label(dialog,text='Export selected images and category objects within them.',padding=8).pack(anchor='w')
        copy=tk.BooleanVar(value=False); ttk.Checkbutton(dialog,text='Copy selected original images into report folder',variable=copy).pack(anchor='w',padx=8)
        ttk.Label(dialog,text='Unchecked: HTML and CSV only; original images are not copied.',padding=8).pack(anchor='w')
        def go():
            folder=filedialog.askdirectory(parent=dialog)
            if not folder:return
            try:
                context = {k:self.vars[k].get() for k in ('category','bin','mode','scene','split')}
                context['ΔY'] = self.brightness_filter if context['mode'] != 'Without category' else None
                export_report(self.ds,folder,self.vars['category'].get(),self.filtered(),self.edges,copy.get(),self.image_rows,context)
                dialog.destroy(); messagebox.showinfo('Export',f'Report saved to {folder}')
            except Exception as e:messagebox.showerror('Export failed',str(e),parent=dialog)
        ttk.Button(dialog,text='Choose output folder and export',command=go).pack(pady=8)

if __name__=='__main__': App().mainloop()



