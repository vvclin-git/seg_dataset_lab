"""Local COCO dataset inspector. Original files are never modified."""
import csv
import hashlib
import html
import json
import math
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
BINS = ('<8', '8–<16', '16–<32', '≥32')

def size_bin(d):
    return BINS[0 if d < 8 else 1 if d < 16 else 2 if d < 32 else 3]

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
    def __init__(self, root, gt, splits, patterns, regex_target='name', work=(960,768)):
        self.root = Path(root).resolve(); self.work = work
        self.images = {}; self.objects = []; self.issues = []; self.categories = {}
        self.by_name = defaultdict(list)
        for p in sorted(self.root.rglob('*')):
            if p.is_file() and p.suffix.lower() in EXTS:
                rel = p.relative_to(self.root).as_posix(); self.by_name[p.name].append(rel)
                self.images[rel] = {'path':p, 'gt':False, 'split':'unassigned', 'groups':{}, 'matches':[], 'annotations':[]}
        rules = [re.compile(p) for p in patterns if p.strip()]
        for rel, item in self.images.items():
            subject = Path(rel).name if regex_target == 'name' else rel
            matches = [(n, m) for n, r in enumerate(rules, 1) if (m := r.search(subject))]
            item['matches'] = [n for n,_ in matches]
            if matches: item['groups'] = matches[0][1].groupdict()
            elif rules: self.issues.append(('regex unmatched', rel))
            if len(matches)>1: self.issues.append(('regex multiple', f'{rel}: {item["matches"]}'))
        gtpath = Path(gt) if gt else None
        files = sorted(gtpath.rglob('*.json')) if gtpath and gtpath.is_dir() else [gtpath] if gtpath and gtpath.is_file() else []
        for fp in files:
            try: self.load_gt(fp)
            except Exception as e: self.issues.append(('invalid GT', f'{fp}: {e}'))
        assignments = defaultdict(set)
        for split, filename in splits.items():
            if not filename: continue
            try:
                with open(filename, encoding='utf-8-sig') as f:
                    for line_no, row in enumerate(csv.reader(f),1):
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
        for img in records:
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
            for i, ann in enumerate(ann_by_id.get(img.get('id'), [])):
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

def export_report(ds, out, category, objects):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fields=['id','image','annotation_id','category_id','split','groups','gt_size','actual_size','work_size','bbox','width','height','diagonal','area','bin']
    with (out/'objects.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for o in objects:
            row={k:o.get(k) for k in fields}; row.update(split=ds.images[o['image']]['split'],groups=json.dumps(ds.images[o['image']]['groups'],ensure_ascii=False),work_size=ds.work); writer.writerow(row)
    groups=defaultdict(list)
    for o in objects: groups[(ds.images[o['image']]['groups'].get('scene') or '(ungrouped)',o['bin'])].append(o)
    with (out/'groups.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['scene','bin','objects','images','object_denominator','image_denominator'])
        for (scene,b), vals in sorted(groups.items()): w.writerow([scene,b,len(vals),len({v['image'] for v in vals}),len(objects),len(ds.images)])
    has={o['image'] for o in ds.objects if str(o['category_id'])==str(category)}
    gt=sum(i['gt'] for i in ds.images.values())
    esc=lambda x:html.escape(str(x))
    rows=''.join(f'<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{len(v)}</td><td>{len({o["image"] for o in v})}</td></tr>' for (a,b),v in sorted(groups.items()))
    issues=''.join(f'<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>' for a,b in ds.issues)
    (out/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Dataset report</title><style>body{{font:15px sans-serif;margin:2em}}td,th{{border:1px solid #bbb;padding:.35em;text-align:left}}table{{border-collapse:collapse}}</style><h1>Segmentation dataset report</h1><p>Images: {len(ds.images)}; GT present: {gt}; target present: {len(has)}; annotated without target: {gt-len(has)}; missing GT: {len(ds.images)-gt}.</p><p>Category: {esc(category)}; selected objects: {len(objects)}; working grid: {ds.work[0]}×{ds.work[1]}, direct scaling. GT dimensions may differ from actual images; see issues and CSV. Object and image denominators: {len(objects)}, {len(ds.images)}.</p><h2>Scene and size</h2><table><tr><th>Scene</th><th>Bin</th><th>Objects</th><th>Images</th></tr>{rows}</table><h2>Issues ({len(ds.issues)})</h2><table><tr><th>Type</th><th>Detail</th></tr>{issues}</table><p>Object IDs and complete measurements: objects.csv</p>''',encoding='utf-8')

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Seg Dataset Lab'); self.geometry('1450x900')
        self.ds=None; self.rows=[]; self.photo=None; self.zoom=1.; self.offset=[0,0]; self.drag=None
        self.vars={k:tk.StringVar(value=v) for k,v in {'root':'','gt':'','train':'','val':'','test':'','regex':r'^(?P<scene>.+?)_(?P<camera>camera\d+)_frame_(?P<frame>\d+)','target':'name','width':'960','height':'768','category':'1','bin':'All','scene':'All','split':'All','lookup':''}.items()}
        top=ttk.Frame(self,padding=5); top.pack(fill='x')
        for r,(key,label,isdir) in enumerate([('root','Images folder',True),('gt','GT JSON / folder',False),('train','Train index',False),('val','Val index',False),('test','Test index',False)]):
            ttk.Label(top,text=label).grid(row=r,column=0,sticky='w'); ttk.Entry(top,textvariable=self.vars[key],width=80).grid(row=r,column=1,sticky='ew'); ttk.Button(top,text='Browse',command=lambda k=key,d=isdir:self.choose(k,d)).grid(row=r,column=2)
            if key=='gt': ttk.Button(top,text='Folder',command=lambda:self.choose('gt',True)).grid(row=r,column=3)
        top.columnconfigure(1,weight=1)
        rules=ttk.Frame(self,padding=5); rules.pack(fill='x')
        ttk.Label(rules,text='Regex (multiple rules separated by ;;)').pack(side='left'); ttk.Entry(rules,textvariable=self.vars['regex'],width=50).pack(side='left',fill='x',expand=True)
        ttk.Combobox(rules,textvariable=self.vars['target'],values=['name','path'],width=6,state='readonly').pack(side='left')
        for k in ('width','height'): ttk.Entry(rules,textvariable=self.vars[k],width=5).pack(side='left')
        for title,cmd in [('Scan',self.scan),('Save project',self.save),('Open project',self.open),('Export report',self.export)]: ttk.Button(rules,text=title,command=cmd).pack(side='left')
        filters=ttk.Frame(self,padding=5); filters.pack(fill='x')
        self.boxes={}
        for k,vals in [('category',[]),('bin',['All',*BINS,'8–<16 and height <4']),('scene',['All']),('split',['All','train','val','test','unassigned'])]:
            ttk.Label(filters,text=k.title()).pack(side='left'); c=ttk.Combobox(filters,textvariable=self.vars[k],values=vals,width=19,state='readonly'); c.pack(side='left'); c.bind('<<ComboboxSelected>>',lambda e:self.refresh()); self.boxes[k]=c
        ttk.Label(filters,text='Object ID').pack(side='left'); ttk.Entry(filters,textvariable=self.vars['lookup'],width=17).pack(side='left'); ttk.Button(filters,text='Find',command=self.find).pack(side='left')
        body=ttk.Panedwindow(self,orient='horizontal'); body.pack(fill='both',expand=True)
        left=ttk.Frame(body); right=ttk.Frame(body); body.add(left,weight=2); body.add(right,weight=3)
        self.tree=ttk.Treeview(left,columns=('image','scene','frame','split','size','area'),show='headings',selectmode='browse')
        for col,width in [('image',190),('scene',110),('frame',60),('split',75),('size',110),('area',55)]: self.tree.heading(col,text=col); self.tree.column(col,width=width,anchor='w')
        sb=ttk.Scrollbar(left,orient='vertical',command=self.tree.yview); self.tree.configure(yscrollcommand=sb.set); self.tree.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y'); self.tree.bind('<<TreeviewSelect>>',self.select)
        opts=ttk.Frame(right); opts.pack(fill='x'); self.showbox=tk.BooleanVar(value=True); self.showmask=tk.BooleanVar(value=True)
        for label,var in [('BBox',self.showbox),('GT overlay',self.showmask)]: ttk.Checkbutton(opts,text=label,variable=var,command=self.render).pack(side='left')
        ttk.Label(opts,text='Wheel: zoom; drag: pan; click object to center').pack(side='left')
        self.canvas=tk.Canvas(right,bg='#252525'); self.canvas.pack(fill='both',expand=True); self.canvas.bind('<Configure>',lambda e:self.render()); self.canvas.bind('<MouseWheel>',self.wheel); self.canvas.bind('<Button-4>',lambda e:self.wheel(e,1)); self.canvas.bind('<Button-5>',lambda e:self.wheel(e,-1)); self.canvas.bind('<ButtonPress-1>',self.press); self.canvas.bind('<B1-Motion>',self.pan)
        self.status=tk.StringVar(value='Select dataset and GT, then Scan.'); ttk.Label(self,textvariable=self.status,anchor='w').pack(fill='x')
        self.current=None
    def choose(self,key,directory):
        value=filedialog.askdirectory() if directory else filedialog.askopenfilename()
        if value:self.vars[key].set(value)
    def scan(self):
        try:
            root=self.vars['root'].get(); gt=self.vars['gt'].get(); work=(int(self.vars['width'].get()),int(self.vars['height'].get()))
            if not Path(root).is_dir() or min(work)<=0: raise ValueError('Choose a valid folder and positive working size')
            patterns=[s.strip() for s in self.vars['regex'].get().split(';;') if s.strip()]
            for p in patterns: re.compile(p)
        except Exception as e: messagebox.showerror('Invalid settings',str(e)); return
        splits={k:self.vars[k].get() for k in ('train','val','test')}; target=self.vars['target'].get()
        self.status.set('Scanning in background…'); self.update_idletasks()
        def worker():
            try:
                ds=Dataset(root,gt,splits,patterns,target,work)
                self.after(0,lambda:self.loaded(ds))
            except Exception as e:
                msg=str(e); self.after(0,lambda:messagebox.showerror('Scan failed',msg))
        threading.Thread(target=worker,daemon=True).start()
    def loaded(self,ds):
        self.ds=ds; self.boxes['category']['values']=[str(x) for x in sorted(ds.categories)]; self.boxes['scene']['values']=['All']+sorted({i['groups'].get('scene','') for i in ds.images.values()}-{''})
        self.refresh()
    def filtered(self):
        if not self.ds:return []
        result=[]
        for o in self.ds.objects:
            i=self.ds.images[o['image']]; b=self.vars['bin'].get(); scene=self.vars['scene'].get(); split=self.vars['split'].get()
            if str(o['category_id'])!=self.vars['category'].get():continue
            if b=='8–<16 and height <4' and not (o['bin']==BINS[1] and o['height']<4):continue
            if b not in ('All','8–<16 and height <4') and o['bin']!=b:continue
            if scene!='All' and i['groups'].get('scene')!=scene:continue
            if split!='All' and i['split']!=split:continue
            result.append(o)
        def sort(o):
            g=self.ds.images[o['image']]['groups']; f=g.get('frame') or ''; return (g.get('scene') or '',g.get('camera') or '',g.get('run') or '',g.get('segment') or '',int(f) if f.isdigit() else -1,o['image'],str(o['annotation_id']))
        return sorted(result,key=sort)
    def refresh(self):
        if not self.ds:return
        self.tree.delete(*self.tree.get_children()); self.rows=self.filtered()
        for n,o in enumerate(self.rows):
            i=self.ds.images[o['image']]; g=i['groups']; self.tree.insert('', 'end',iid=str(n),values=(o['image'],g.get('scene',''),g.get('frame',''),i['split'],f'{o["width"]:.1f}×{o["height"]:.1f} / {o["diagonal"]:.1f}',o['area'] if o['area'] is not None else '—'))
        self.status.set(f'{len(self.ds.images)} images | {sum(i["gt"] for i in self.ds.images.values())} with GT | {len(self.rows)} matching objects in {len({o["image"] for o in self.rows})} images | {len(self.ds.issues)} issues (export report for details)')
    def find(self):
        for n,o in enumerate(self.rows):
            if o['id']==self.vars['lookup'].get().strip(): self.tree.selection_set(str(n)); self.tree.see(str(n)); self.select(); return
        messagebox.showinfo('Find','Object ID not found in current filters.')
    def select(self,event=None):
        if not self.tree.selection():return
        self.current=self.rows[int(self.tree.selection()[0])]; self.zoom=1.; self.offset=[0,0]; self.render()
    def render(self):
        if not self.ds or not self.current:return
        o=self.current; item=self.ds.images[o['image']]
        try:
            im=Image.open(item['path']).convert('RGB').resize(self.ds.work,Image.Resampling.BILINEAR)
            if self.showmask.get():
                m=mask_for(o['segmentation'],*o['gt_size'])
                if m is not None:
                    m=m.resize(self.ds.work,Image.Resampling.NEAREST); overlay=Image.new('RGB',self.ds.work,(255,30,30)); im=Image.composite(Image.blend(im,overlay,.45),im,m)
            if self.showbox.get():
                x,y,w,h=o['bbox']; sx=self.ds.work[0]/o['gt_size'][0]; sy=self.ds.work[1]/o['gt_size'][1]
                ImageDraw.Draw(im).rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='yellow',width=1)
            cw=max(self.canvas.winfo_width(),1); ch=max(self.canvas.winfo_height(),1); scale=min(cw/im.width,ch/im.height)*self.zoom
            im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.NEAREST if scale>=1 else Image.Resampling.BILINEAR)
            self.photo=ImageTk.PhotoImage(im); self.canvas.delete('all'); self.canvas.create_image(cw/2+self.offset[0],ch/2+self.offset[1],image=self.photo)
            self.canvas.create_text(8,8,anchor='nw',fill='white',text=f'ID {o["id"]}  GT {o["gt_size"]} → work {self.ds.work}  mask area {o["area"]}')
        except Exception as e:self.status.set(f'Preview error: {e}')
    def wheel(self,e,direction=None):
        self.zoom=max(.2,min(30,self.zoom*(1.25 if (direction or e.delta)>0 else .8))); self.render()
    def press(self,e):self.drag=(e.x,e.y)
    def pan(self,e):
        if self.drag:self.offset[0]+=e.x-self.drag[0]; self.offset[1]+=e.y-self.drag[1]; self.drag=(e.x,e.y); self.render()
    def save(self):
        p=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
        if p: Path(p).write_text(json.dumps({'version':1,'settings':{k:v.get() for k,v in self.vars.items()}},indent=2,ensure_ascii=False),encoding='utf-8')
    def open(self):
        p=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not p:return
        try:
            data=json.loads(Path(p).read_text(encoding='utf-8'))
            if data.get('version')!=1:raise ValueError('Unsupported project version')
            for k,v in data['settings'].items():
                if k in self.vars:self.vars[k].set(v)
            self.scan()
        except Exception as e:messagebox.showerror('Project error',str(e))
    def export(self):
        if not self.ds:return
        p=filedialog.askdirectory()
        if p:
            try:export_report(self.ds,p,self.vars['category'].get(),self.filtered()); messagebox.showinfo('Export',f'Report saved to {p}')
            except Exception as e:messagebox.showerror('Export failed',str(e))

if __name__=='__main__': App().mainloop()
