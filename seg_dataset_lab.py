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

class Dataset:
    def __init__(self, root, gt, splits, patterns, regex_target='name', work=(960,768), progress=None):
        self.root = Path(root).resolve(); self.work = work
        self.progress = progress or (lambda *args: None)
        self.progress('Discovering images', None, None)
        self.images = {}; self.objects = []; self.issues = []; self.categories = {}
        self.by_name = defaultdict(list)
        for count, p in enumerate(self.root.rglob('*'), 1):
            if count % 250 == 0: self.progress(f'Discovering files: {count:,}', None, None)
            if p.is_file() and p.suffix.lower() in EXTS:
                rel = p.relative_to(self.root).as_posix(); self.by_name[p.name].append(rel)
                self.images[rel] = {'path':p, 'gt':False, 'split':'unassigned', 'groups':{}, 'matches':[], 'annotations':[]}
        rules = [re.compile(p) for p in patterns if p.strip()]
        for index, (rel, item) in enumerate(self.images.items(), 1):
            if index == 1 or index % 100 == 0 or index == len(self.images): self.progress('Grouping images', index, len(self.images))
            subject = Path(rel).name if regex_target == 'name' else rel
            matches = [(n, m) for n, r in enumerate(rules, 1) if (m := r.search(subject))]
            item['matches'] = [n for n,_ in matches]
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
            if not item['gt']: self.issues.append(('missing GT', rel))

    def resolve(self, value):
        name = normalize(value)
        if name in self.images: return name
        candidates = self.by_name.get(Path(name).name, [])
        return candidates[0] if len(candidates)==1 else None

    def load_gt(self, fp):
        data = json.loads(fp.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or not ('images' in data or 'image' in data) or not isinstance(data.get('annotations'), list):
            raise ValueError('not a COCO annotation document')
        for c in data.get('categories', []): self.categories[c['id']] = c.get('name', str(c['id']))
        records = data.get('images', [data.get('image')]); ann_by_id = defaultdict(list)
        for ann in data['annotations']: ann_by_id[ann.get('image_id')].append(ann)
        for record_index, img in enumerate(records, 1):
            if len(records) > 1 and (record_index == 1 or record_index % 20 == 0): self.progress(f'GT {fp.name}: images', record_index, len(records))
            if not isinstance(img, dict): continue
            rel = self.resolve(img.get('file_name',''))
            if rel is None:
                self.issues.append(('GT image missing or ambiguous', f'{fp}: {img.get("file_name")}')); continue
            item = self.images[rel]
            if item['gt']:
                self.issues.append(('duplicate GT', f'{rel}: {fp}')); continue
            try:
                w, h = int(img['width']), int(img['height'])
                if w <= 0 or h <= 0: raise ValueError('invalid dimensions')
                with Image.open(item['path']) as im: actual = im.size
            except Exception as e:
                self.issues.append(('invalid image dimensions', f'{rel}: {e}')); continue
            item.update(gt=True, gt_size=(w,h), actual_size=actual, source=str(fp), image_id=img.get('id'))
            if actual != (w,h): self.issues.append(('dimension mismatch (scaled)', f'{rel}: GT {w}x{h}, image {actual[0]}x{actual[1]}'))
            anns = ann_by_id.get(img.get('id'), [])
            for i, ann in enumerate(anns):
                if i == 0 or i % 20 == 0: self.progress(f'GT {fp.name}: annotations', i, len(anns))
                try:
                    x,y,bw,bh = map(float, ann['bbox'])
                    if bw < 0 or bh < 0: raise ValueError('negative bbox')
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
                    obj['bin']=size_bin(obj['diagonal']); self.objects.append(obj); item['annotations'].append(obj)
                except (KeyError, TypeError, ValueError) as e: self.issues.append(('invalid annotation', f'{rel}: {e}'))

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


def match_details(item, category, bin_name, edges, mode='Any'):
    objects = [o for o in item['annotations'] if category == 'All' or str(o['category_id']) == str(category)]
    total = len(objects)
    count = sum(object_matches_bin(o, bin_name, edges) for o in objects)
    if mode == 'Without category':
        return bool(item.get('gt')) and total == 0, count, total, False
    if category == 'All' and bin_name == 'All' and mode == 'Any':
        return True, count, total, False
    if not total: return False, count, total, False
    if mode == 'All': return count == total, count, total, False
    if mode == 'Mode':
        if bin_name == 'All': return True, count, total, False
        buckets = Counter()
        for obj in objects:
            # Keep the special sub-bin disjoint from all ordinary bins.
            label = SPECIAL_BIN if bin_name == SPECIAL_BIN and object_matches_bin(obj, SPECIAL_BIN, edges) else size_bin(obj['diagonal'], edges)
            buckets[label] += 1
        maximum = max(buckets.values())
        return count > 0 and count == maximum, count, total, count > 0 and count == maximum and sum(v == maximum for v in buckets.values()) > 1
    if mode == 'Any': return count > 0, count, total, False
    raise ValueError(f'Unknown filter mode: {mode}')


def filter_images(ds, category, bin_name, scene, split, edges, mode='Any'):
    selected = []
    for rel, item in ds.images.items():
        if scene != 'All' and item.get('scene', '(unmatched)') != scene: continue
        if split != 'All' and item['split'] != split: continue
        if match_details(item, category, bin_name, edges, mode)[0]: selected.append(rel)
    return selected


def objects_in_images(ds, image_paths, category):
    return [o for rel in image_paths for o in ds.images[rel]['annotations']
            if category == 'All' or str(o['category_id']) == str(category)]


def export_report(ds, out, category, objects, edges=(8,16,32), copy_images=False, image_paths=None, filter_context=None):
    image_paths = list({o['image'] for o in objects} if image_paths is None else image_paths)
    out = Path(out).resolve()
    if out == ds.root or ds.root in out.parents:
        raise ValueError('Choose a report folder outside the image dataset root to protect source files.')
    out.mkdir(parents=True, exist_ok=True)
    fields=['id','image','annotation_id','category_id','split','groups','gt_size','actual_size','work_size','bbox','width','height','diagonal','area','bin']
    with (out/'objects.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for o in objects:
            row={k:o.get(k) for k in fields}; row.update(split=ds.images[o['image']]['split'],groups=json.dumps({**ds.images[o['image']]['groups'], 'scene':ds.images[o['image']].get('scene','(unmatched)')},ensure_ascii=False),work_size=ds.work,bin=size_bin(o['diagonal'],edges)); writer.writerow(row)
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
    gt=sum(ds.images[rel]['gt'] for rel in image_paths)
    esc=lambda x:html.escape(str(x))
    rows=''.join(f'<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{len(v)}</td><td>{len({o["image"] for o in v})}</td></tr>' for (a,b),v in sorted(groups.items()))
    issues=''.join(f'<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>' for a,b in ds.issues)
    links=''.join(f'<li><a href="images/{quote(rel, safe="/")}">{esc(rel)}</a></li>' for rel in copied)
    context_text = html.escape(json.dumps(filter_context or {}, ensure_ascii=False))
    image_text=f'Copied {len(copied)} selected images.<ul>{links}</ul>' if copy_images else 'No images copied. Object paths are relative to the original dataset root.'
    (out/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Dataset report</title><style>body{{font:15px sans-serif;margin:2em}}td,th{{border:1px solid #bbb;padding:.35em;text-align:left}}table{{border-collapse:collapse}}</style><h1>Segmentation dataset report</h1><p>Image filters: {context_text}</p><p>Selected images: {len(image_paths)}; GT present: {gt}; target present: {len(has)}; annotated without target: {gt-len(has)}; missing GT: {len(image_paths)-gt}.</p><p>Category: {esc(category)}; selected objects: {len(objects)}; working grid: {ds.work[0]}×{ds.work[1]}, direct scaling; diagonal bin edges: {esc(edges)} px. Object and image denominators: {len(objects)}, {len(image_paths)}.</p><p>{image_text}</p><h2>Scene and size</h2><table><tr><th>Scene</th><th>Bin</th><th>Objects</th><th>Images</th></tr>{rows}</table><h2>Issues ({len(ds.issues)})</h2><table><tr><th>Type</th><th>Detail</th></tr>{issues}</table><p>Object IDs and complete measurements: objects.csv</p>''',encoding='utf-8')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Seg Dataset Lab'); self.geometry('1500x900')
        self.ds=None; self.rows=[]; self.image_rows=[]; self.object_rows=[]
        self.current_image=None; self.current=None; self.photo=None
        self.zoom=1.; self.offset=[0,0]; self.drag=None; self.events=queue.Queue(); self.scanning=False
        self.edges=[8.,16.,32.]; self.chart=None; self.chart_bars=[]
        self.scene_rules=[]; self.sort_column='image'; self.sort_desc=False
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
        body=ttk.Panedwindow(self,orient='horizontal'); body.pack(fill='both',expand=True)
        left=ttk.Panedwindow(body,orient='horizontal'); right=ttk.Frame(body); body.add(left,weight=3); body.add(right,weight=3)
        images=ttk.Frame(left); objects=ttk.Frame(left); left.add(images,weight=1); left.add(objects,weight=1)
        ttk.Label(images,text='Images (select to view)',font=('TkDefaultFont',10,'bold')).pack(anchor='w')
        self.image_tree=self.make_tree(images,[('image',200),('scene',75),('split',75),('GT',65),('match',75)])
        self.image_tree.bind('<<TreeviewSelect>>',self.select_image)
        for col in ('image','scene','split','GT','match'): self.image_tree.heading(col,command=lambda c=col:self.sort_images(c))
        ttk.Label(objects,text='GT objects in selected image',font=('TkDefaultFont',10,'bold')).pack(anchor='w')
        self.object_tree=self.make_tree(objects,[('id',105),('class',65),('size',125),('area',60)])
        self.object_tree.bind('<<TreeviewSelect>>',self.select_object)
        opts=ttk.Frame(right); opts.pack(fill='x'); self.showbox=tk.BooleanVar(value=True); self.showmask=tk.BooleanVar(value=True)
        for label,var in [('BBox',self.showbox),('GT overlay',self.showmask)]: ttk.Checkbutton(opts,text=label,variable=var,command=self.render).pack(side='left')
        ttk.Label(opts,text='Mouse wheel: zoom; drag: pan').pack(side='left')
        self.canvas=tk.Canvas(right,bg='#252525'); self.canvas.pack(fill='both',expand=True); self.canvas.bind('<Configure>',lambda e:self.render()); self.canvas.bind('<MouseWheel>',self.wheel); self.canvas.bind('<Button-4>',lambda e:self.wheel(e,1)); self.canvas.bind('<Button-5>',lambda e:self.wheel(e,-1)); self.canvas.bind('<ButtonPress-1>',self.press); self.canvas.bind('<B1-Motion>',self.pan)
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
        except Exception as e: messagebox.showerror('Invalid settings',str(e)); return
        splits={k:self.vars[k].get() for k in ('train','val','test')}; target=self.vars['target'].get()
        self.scanning=True; self.scan_button.state(['disabled']); self.progress.configure(mode='indeterminate'); self.progress.start(12); self.status.set('Scanning: discovering images…')
        def worker():
            last=[0.]
            def progress(phase, current, total):
                now=time.monotonic()
                if now-last[0] > .1 or current == total:
                    self.events.put(('progress',(phase,current,total))); last[0]=now
            try:self.events.put(('done',Dataset(root,gt,splits,patterns,target,work,progress)))
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
        self.ds=ds; self.current=None; self.current_image=None; self.canvas.delete('all')
        self.boxes['category']['values']=['All']+[str(x) for x in sorted(ds.categories)]
        self.apply_scene_rules()

    def apply_scene_rules(self):
        assign_scenes(self.ds.images,self.scene_rules,self.vars['target'].get())
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
        self.image_rows=filter_images(self.ds,self.vars['category'].get(),self.vars['bin'].get(),self.vars['scene'].get(),self.vars['split'].get(),self.edges,self.vars['mode'].get())
        def sort_key(rel):
            item=self.ds.images[rel]
            details=match_details(item,self.vars['category'].get(),self.vars['bin'].get(),self.edges,self.vars['mode'].get())
            keys={'image':rel.casefold(),'scene':item['scene'].casefold(),'split':item['split'].casefold(),'GT':len(item['annotations']) if item['gt'] else -1,'match':(details[1],details[2]) if item['gt'] else (-1,-1)}
            return keys[self.sort_column]
        self.image_rows.sort(key=sort_key,reverse=self.sort_desc)
        self.rows=self.filtered()
        self.image_tree.delete(*self.image_tree.get_children()); self.object_tree.delete(*self.object_tree.get_children()); self.object_rows=[]
        for n,rel in enumerate(self.image_rows):
            i=self.ds.images[rel]; state=f'{len(i["annotations"])} GT' if i['gt'] else 'missing'
            _,matched,total,tied=match_details(i,self.vars['category'].get(),self.vars['bin'].get(),self.edges,self.vars['mode'].get())
            match_text=f'{matched}/{total}'+(' tie' if tied else '') if i['gt'] else '—'
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
        self.current_image=rel; self.current=None; self.zoom=1.; self.offset=[0,0]; self.populate_objects(); self.render()
    def populate_objects(self):
        self.object_tree.delete(*self.object_tree.get_children())
        if not self.current_image:return
        matching={o['id'] for o in self.rows if o['image']==self.current_image}
        self.object_rows=list(self.ds.images[self.current_image]['annotations'])
        for n,o in enumerate(self.object_rows):
            cls=self.ds.categories.get(o['category_id'],str(o['category_id'])); tag='match' if o['id'] in matching else 'other'
            self.object_tree.insert('','end',iid=str(n),values=(o['id'],cls,f'{o["width"]:.1f}×{o["height"]:.1f} / {o["diagonal"]:.1f}',o['area'] if o['area'] is not None else '—'),tags=(tag,))
        self.object_tree.tag_configure('other',foreground='#888888')
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
    def render(self):
        if not self.ds or not self.current_image:return
        o=self.current; item=self.ds.images[self.current_image]
        try:
            with Image.open(item['path']) as source: im=source.convert('RGB').resize(self.ds.work,Image.Resampling.BILINEAR)
            if o and self.showmask.get():
                m=mask_for(o['segmentation'],*o['gt_size'])
                if m is not None:
                    m=m.resize(self.ds.work,Image.Resampling.NEAREST); overlay=Image.new('RGB',self.ds.work,(255,30,30)); im=Image.composite(Image.blend(im,overlay,.45),im,m)
            if o and self.showbox.get():
                x,y,w,h=o['bbox']; sx=self.ds.work[0]/o['gt_size'][0]; sy=self.ds.work[1]/o['gt_size'][1]
                ImageDraw.Draw(im).rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='yellow',width=1)
            cw=max(self.canvas.winfo_width(),1); ch=max(self.canvas.winfo_height(),1); scale=min(cw/im.width,ch/im.height)*self.zoom
            im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.NEAREST if scale>=1 else Image.Resampling.BILINEAR)
            self.photo=ImageTk.PhotoImage(im); self.canvas.delete('all'); self.canvas.create_image(cw/2+self.offset[0],ch/2+self.offset[1],image=self.photo)
            label=f'ID {o["id"]}  GT {o["gt_size"]} → work {self.ds.work}  mask area {o["area"]}' if o else f'{self.current_image} — select a GT object to overlay'
            self.canvas.create_text(8,8,anchor='nw',fill='white',text=label)
        except Exception as e:self.status.set(f'Preview error: {e}')
    def wheel(self,e,direction=None):
        self.zoom=max(.2,min(30,self.zoom*(1.25 if (direction or e.delta)>0 else .8))); self.render()
    def press(self,e):self.drag=(e.x,e.y)
    def pan(self,e):
        if self.drag:self.offset[0]+=e.x-self.drag[0]; self.offset[1]+=e.y-self.drag[1]; self.drag=(e.x,e.y); self.render()
    def apply_edges(self):
        try:edges=parse_edges(self.vars['edges'].get())
        except ValueError as e:messagebox.showerror('Invalid bins',str(e),parent=self.chart or self); return
        self.edges=edges; self.vars['bin'].set('All'); self.boxes['bin']['values']=['All',*bin_labels(edges),'8–<16 and height <4']; self.refresh()
    def show_chart(self):
        if not self.ds:messagebox.showinfo('Statistics','Scan a dataset first.'); return
        if self.chart and self.chart.winfo_exists():self.chart.lift(); return
        self.chart=tk.Toplevel(self); self.chart.title('Object size distribution'); self.chart.geometry('900x540')
        controls=ttk.Frame(self.chart,padding=6); controls.pack(fill='x')
        ttk.Label(controls,text='Diagonal bin edges (px, comma-separated)').pack(side='left'); ttk.Entry(controls,textvariable=self.vars['edges'],width=25).pack(side='left'); ttk.Button(controls,text='Apply bins',command=self.apply_edges).pack(side='left')
        self.chart_note=ttk.Label(self.chart,text='Counts include category objects in filtered images, including objects outside the selected bin. Click a bar to filter images.'); self.chart_note.pack(anchor='w',padx=6)
        self.chart_canvas=tk.Canvas(self.chart,bg='white'); self.chart_canvas.pack(fill='both',expand=True,padx=8,pady=8)
        self.chart_canvas.bind('<Configure>',lambda e:self.draw_chart()); self.chart_canvas.bind('<Button-1>',self.chart_click)
        self.draw_chart()
    def draw_chart(self):
        if not self.ds or not self.chart or not self.chart.winfo_exists():return
        c=self.chart_canvas; c.delete('all'); self.chart_bars=[]
        objs=self.filtered(); labels=bin_labels(self.edges); counts=Counter(size_bin(o['diagonal'],self.edges) for o in objs)
        w=max(c.winfo_width(),300); h=max(c.winfo_height(),200); left=55; bottom=h-55; top=30; plotw=w-left-20; maxcount=max([counts[x] for x in labels]+[1]); step=plotw/len(labels)
        c.create_line(left,top,left,bottom,fill='#555'); c.create_line(left,bottom,w-20,bottom,fill='#555'); c.create_text(8,top,anchor='nw',text=str(maxcount))
        for n,label in enumerate(labels):
            x1=left+n*step+step*.12; x2=left+(n+1)*step-step*.12; barheight=(bottom-top-18)*counts[label]/maxcount; y=bottom-barheight
            color='#225eaa' if self.vars['bin'].get()==label else '#4e9acb'; c.create_rectangle(x1,y,x2,bottom,fill=color,outline='')
            c.create_text((x1+x2)/2,max(top,y-14),text=str(counts[label])); c.create_text((x1+x2)/2,bottom+15,text=label)
            self.chart_bars.append((x1,x2,label))
        c.create_text(w/2,h-8,text=f'BBox diagonal (px) • {len(objs):,} objects • {len(self.image_rows):,} selected images',fill='#444')
    def chart_click(self,event):
        for x1,x2,label in self.chart_bars:
            if x1<=event.x<=x2:self.vars['bin'].set(label); self.refresh(); return
    def save(self):
        p=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
        if p:
            settings={k:v.get() for k,v in self.vars.items()}; settings['edges']=','.join(f'{x:g}' for x in self.edges)
            Path(p).write_text(json.dumps({'version':1,'settings':settings,'scene_rules':self.scene_rules},indent=2,ensure_ascii=False),encoding='utf-8')
    def open(self):
        p=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not p:return
        try:
            data=json.loads(Path(p).read_text(encoding='utf-8'))
            if data.get('version')!=1:raise ValueError('Unsupported project version')
            self.vars['mode'].set('Any')
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
            try:export_report(self.ds,folder,self.vars['category'].get(),self.filtered(),self.edges,copy.get(),self.image_rows,{k:self.vars[k].get() for k in ('category','bin','mode','scene','split')}); dialog.destroy(); messagebox.showinfo('Export',f'Report saved to {folder}')
            except Exception as e:messagebox.showerror('Export failed',str(e),parent=dialog)
        ttk.Button(dialog,text='Choose output folder and export',command=go).pack(pady=8)

if __name__=='__main__': App().mainloop()


