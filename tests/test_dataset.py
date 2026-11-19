import csv
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from seg_dataset_lab import Dataset, export_report, parse_edges, bin_labels, size_bin

class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'data'
        self.root.mkdir()
        Image.new('RGB', (20, 10), 'white').save(self.root / 'a.png')
        Image.new('RGB', (20, 10), 'white').save(self.root / 'b.png')
        self.gt = self.base / 'gt.json'
        self.gt.write_text(json.dumps({'images':[{'id':1,'file_name':'a.png','width':40,'height':20}], 'categories':[{'id':1,'name':'boat'}], 'annotations':[{'id':9,'image_id':1,'category_id':1,'bbox':[4,2,8,4],'segmentation':[[4,2,12,2,12,6,4,6]]}]}))
        events=[]
        self.ds=Dataset(self.root,self.gt,{},[],work=(20,10),progress=lambda *x:events.append(x))
        self.assertTrue(events)
    def tearDown(self):self.temp.cleanup()
    def test_bins(self):
        edges=parse_edges('5,10,20')
        self.assertEqual([size_bin(n,edges) for n in (4,5,10,20)],bin_labels(edges))
        for value in ('','0,5','10,5','nan','2,2'):
            with self.assertRaises(ValueError):parse_edges(value)
    def test_mapping_and_report_modes(self):
        self.assertEqual(len(self.ds.images),2)
        obj=self.ds.objects[0]
        self.assertEqual((obj['width'],obj['height']),(4,2))
        self.assertFalse(self.ds.images['b.png']['gt'])
        no_copy=self.base/'without'; export_report(self.ds,no_copy,1,[obj],[2,5],False)
        self.assertFalse((no_copy/'images').exists())
        with (no_copy/'objects.csv').open(encoding='utf-8-sig',newline='') as f:self.assertEqual(next(csv.DictReader(f))['bin'],'2–<5')
        with_copy=self.base/'with'; export_report(self.ds,with_copy,1,[obj],[2,5],True)
        self.assertEqual((with_copy/'images'/'a.png').read_bytes(),(self.root/'a.png').read_bytes())
        self.assertFalse((with_copy/'images'/'b.png').exists())
        with self.assertRaises(ValueError):export_report(self.ds,self.root/'report',1,[obj])

if __name__=='__main__':unittest.main()


class FilteringSceneTest(unittest.TestCase):
    def test_scene_matching_and_image_filter_scope(self):
        from seg_dataset_lab import assign_scenes, filter_images, objects_in_images
        from types import SimpleNamespace
        def obj(category, diagonal):return {'category_id':category,'diagonal':diagonal,'height':2}
        images={
            'cam_a_1.png': {'groups':{'scene':'alpha'},'annotations':[obj(1,6),obj(1,24),obj(2,6)],'split':'train'},
            'cam_b_2.png': {'groups':{'scene':'beta'},'annotations':[obj(1,24)],'split':'val'},
            'other.png': {'groups':{},'annotations':[],'split':'unassigned'},
        }
        ds=SimpleNamespace(images=images)
        assign_scenes(images,[('',r'^cam_a'),('Harbor',r'^cam_'),('B',r'^cam_b')])
        self.assertEqual(images['cam_a_1.png']['scene'],'A')
        self.assertEqual(images['cam_b_2.png']['scene'],'Harbor')
        self.assertEqual(len(images['cam_b_2.png']['scene_matches']),2)
        self.assertEqual(images['other.png']['scene'],'(unmatched)')
        selected=filter_images(ds,'1','<8','All','All',[8,16,32])
        self.assertEqual(selected,['cam_a_1.png'])
        self.assertEqual([o['diagonal'] for o in objects_in_images(ds,selected,'1')],[6,24])
        self.assertEqual(filter_images(ds,'All','All','All','All',[8,16,32]),list(images))
        self.assertEqual(filter_images(ds,'1','<8','All','val',[8,16,32]),[])
        assign_scenes(images,[])
        self.assertEqual([images[x]['scene'] for x in images],['A','B','(unmatched)'])


class GTStatusTest(unittest.TestCase):
    def test_uncertain_gt_is_not_absence_and_is_exported(self):
        from seg_dataset_lab import gt_status, filter_images, objects_in_images
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'images'; root.mkdir(); gt=base/'gt'; gt.mkdir()
            for name in ('valid','invalid','duplicate','missing','orphan','no_category'):
                Image.new('RGB',(20,10)).save(root/f'{name}.png')
            def write(name, image, annotations, **extra):
                (gt/name).write_text(json.dumps({'images':[{'id':1,'file_name':image+'.png','width':20,'height':10}], 'annotations':annotations, **extra}))
            write('a.json','valid',[])
            write('b.json','invalid',[{'image_id':1,'bbox':[0,0,float('nan'),1]}])
            write('c.json','duplicate',[])
            write('d.json','duplicate',[{'image_id':1,'category_id':1,'bbox':[0,0,2,2]}])
            write('e.json','orphan',[{'image_id':99,'category_id':1,'bbox':[0,0,2,2]}])
            write('f.json','no_category',[{'image_id':1,'bbox':[0,0,2,2]}])
            ds=Dataset(root,gt,{},[])
            self.assertEqual({name:gt_status(ds.images[name+'.png']) for name in ('valid','invalid','duplicate','missing','orphan','no_category')},
                {'valid':'valid','invalid':'invalid','duplicate':'duplicate','missing':'missing','orphan':'invalid','no_category':'invalid'})
            self.assertEqual(filter_images(ds,'1','All','All','All',[8,16,32],'Without category'),['valid.png'])
            paths=list(ds.images); out=base/'report'
            export_report(ds,out,'1',objects_in_images(ds,paths,'1'),image_paths=paths)
            with (out/'images.csv').open(encoding='utf-8-sig',newline='') as f: rows={r['image']:r for r in csv.DictReader(f)}
            self.assertEqual(len(rows),6)
            self.assertEqual(len(json.loads(rows['duplicate.png']['gt_sources'])),2)
            self.assertEqual(rows['invalid.png']['gt_status'],'invalid')
            self.assertIn('valid gt without target: 1', (out/'report.html').read_text(encoding='utf-8').lower())


class MatchModesTest(unittest.TestCase):
    def test_any_all_mode_tie_and_missing(self):
        from seg_dataset_lab import match_details
        obj=lambda size: {'category_id':1,'diagonal':size,'height':2}
        item={'gt':True,'annotations':[obj(10),obj(11),obj(12),obj(50),obj(51)]}
        edges=[8,16,32]
        self.assertEqual(match_details(item,'1','8–<16',edges,'Any'),(True,3,5,False))
        self.assertEqual(match_details(item,'1','8–<16',edges,'All'),(False,3,5,False))
        self.assertEqual(match_details(item,'1','8–<16',edges,'Mode'),(True,3,5,False))
        self.assertFalse(match_details(item,'1','≥32',edges,'Mode')[0])
        item['annotations'].pop(2)
        self.assertEqual(match_details(item,'1','8–<16',edges,'Mode'),(True,2,4,True))
        self.assertEqual(match_details(item,'1','≥32',edges,'Mode'),(True,2,4,True))
        item['annotations']=[]
        for mode in ('Any','All','Mode'):
            self.assertFalse(match_details(item,'1','<8',edges,mode)[0])
        self.assertTrue(match_details(item,'1','<8',edges,'Without category')[0])
        item['gt']=False
        self.assertFalse(match_details(item,'1','<8',edges,'Without category')[0])
    def test_special_bin_mode_and_all_category(self):
        from seg_dataset_lab import match_details, SPECIAL_BIN
        item={'gt':True,'annotations':[{'category_id':1,'diagonal':10,'height':2}, {'category_id':1,'diagonal':10,'height':5}, {'category_id':2,'diagonal':10,'height':5}]}
        self.assertEqual(match_details(item,'1',SPECIAL_BIN,[8,16,32],'Mode'),(True,1,2,True))
        self.assertFalse(match_details(item,'All',SPECIAL_BIN,[8,16,32],'Mode')[0])
        self.assertTrue(match_details(item,'1','All',[8,16,32],'All')[0])


class PreviewTest(unittest.TestCase):
    def test_coordinates_with_pan_and_zoom(self):
        from seg_dataset_lab import preview_to_work
        self.assertEqual(preview_to_work(150,150,(400,400),(200,200),(50,50),(100,100)),(0,0))
        self.assertEqual(preview_to_work(249,249,(400,400),(200,200),(50,50),(100,100)),(49,49))
        self.assertIsNone(preview_to_work(149,150,(400,400),(200,200),(50,50),(100,100)))
    def test_click_prefers_smaller_mask_and_sort_numeric(self):
        import numpy as np
        from seg_dataset_lab import hit_mask, object_sort_key
        big={'id':'1','category_id':1,'diagonal':100,'area':100}
        small={'id':'2','category_id':1,'diagonal':9,'area':9}
        m1=np.ones((4,4),dtype=bool); m2=np.zeros((4,4),dtype=bool); m2[1,1]=True
        self.assertIs(hit_mask([big,small],{id(big):m1,id(small):m2},1,1),small)
        self.assertIs(hit_mask([big,small],{id(big):m1,id(small):m2},2,2),big)
        self.assertEqual(sorted([big,small],key=lambda o:object_sort_key(o,'size',{})),[small,big])
        self.assertEqual(sorted([big,small],key=lambda o:object_sort_key(o,'area',{})),[small,big])


class BrightnessTest(unittest.TestCase):
    def test_signed_difference_exclusions_and_sample_limits(self):
        import numpy as np
        from seg_dataset_lab import measure_brightness

        target = np.zeros((40, 40), dtype=bool)
        target[:6, :6] = True
        excluded = np.zeros_like(target)
        excluded[:18, 6:18] = True
        y = np.full(target.shape, 100.0)
        y[excluded] = 255
        y[target] = 60
        result = measure_brightness(y, target, excluded)
        self.assertEqual(result['delta_y'], 40)
        self.assertEqual(result['object_pixels'], 36)
        self.assertEqual(result['background_pixels'], 72)
        y[target] = 140
        self.assertEqual(measure_brightness(y, target, excluded)['delta_y'], -40)
        self.assertEqual(measure_brightness(y, target, ~target)['brightness_status'], 'insufficient_background_pixels')
        target[1:, :] = False
        self.assertEqual(measure_brightness(y, target, excluded)['brightness_status'], 'insufficient_object_pixels')
        y[~target & ~excluded] = 100
        adjusted = measure_brightness(y, target, excluded, settings=(2, 6, 1))
        self.assertEqual(adjusted['delta_y'], -40)

    def test_dataset_measurement_exports_signed_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'images'
            root.mkdir()
            image = Image.new('RGB', (40, 40), (100, 100, 100))
            image.paste((60, 60, 60), (10, 10, 16, 16))
            image.save(root / 'boat.png')
            gt = base / 'gt.json'
            gt.write_text(json.dumps({
                'images': [{'id': 1, 'file_name': 'boat.png', 'width': 40, 'height': 40}],
                'annotations': [
                    {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 6, 6],
                     'segmentation': [[10, 10, 15, 10, 15, 15, 10, 15]]},
                    {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [29, 29, 5, 5]},
                ],
            }), encoding='utf-8')
            from seg_dataset_lab import assign_scenes
            ds = Dataset(root, gt, {}, [r'^other'], work=(40, 40))
            assign_scenes(ds.images, [('Harbor', r'^boat')])
            ds.analyze_brightness((10, 32, 64))
            out = base / 'report'
            export_report(ds, out, '1', ds.objects, image_paths=['boat.png'])
            with (out / 'objects.csv').open(encoding='utf-8-sig', newline='') as handle:
                measured, unknown = csv.DictReader(handle)
            self.assertEqual(measured['brightness_status'], 'measured')
            self.assertAlmostEqual(float(measured['delta_y']), 40)
            self.assertEqual(unknown['brightness_status'], 'mask_unavailable')
            self.assertEqual(unknown['delta_y'], '')
            with (out / 'images.csv').open(encoding='utf-8-sig', newline='') as handle:
                row, = csv.DictReader(handle)
            self.assertEqual(row['regex_status'], 'Scene rule matched')
            self.assertEqual(row['grouping_rule_hits'], '[]')
            self.assertEqual(row['measured_objects'], '1')
            report = (out / 'report.html').read_text(encoding='utf-8')
            self.assertIn('Ring radius 10 px', report)
            self.assertIn('<td>Median |ΔY|</td><td>40.0</td>', report)
            self.assertIn('Absolute |ΔY| histogram', report)
            self.assertIn('Valid GT with target: 1', report)
            self.assertIn('Scene rule matched', report)
            self.assertIn((root / 'boat.png').as_uri(), report)
            self.assertNotIn('regex unmatched', report)

    def test_delta_filter_combines_size_category_and_brightness_on_one_object(self):
        from seg_dataset_lab import filter_images
        from types import SimpleNamespace

        def item(*objects):
            return {'annotations': objects}

        def obj(diagonal, delta, category_id=1):
            return {'category_id': category_id, 'diagonal': diagonal, 'delta_y': delta}

        ds = SimpleNamespace(images={
            'mixed': item(obj(4, 40), obj(12, -15)),
            'negative': item(obj(4, -15)),
            'positive': item(obj(4, 15)),
            'zero': item(obj(4, 0)),
            'unknown': item(obj(4, None)),
            'different': item(obj(4, -15, 2)),
        })
        for kind, lower, upper, expected in [
            ('Negative', -15, 0, ['negative']),
            ('Positive', 0, 15, ['positive']),
            ('Absolute', 0, 15, ['negative','positive','zero']),
            ('Signed', -15, 15, ['negative','positive','zero']),
        ]:
            with self.subTest(kind=kind):
                self.assertEqual(filter_images(ds,'1','<8','All','All',(8,16,32),
                                               brightness=(kind,lower,upper)), expected)


class ObjectRowFilterTest(unittest.TestCase):
    def test_populate_objects_uses_active_category_bin_and_mode(self):
        from types import SimpleNamespace
        from seg_dataset_lab import App, SPECIAL_BIN, objects_in_images
        class Var:
            def __init__(self, value): self.value=value
            def get(self): return self.value
        class Tree:
            def __init__(self): self.rows=[]
            def get_children(self): return []
            def delete(self, *args): self.rows=[]
            def insert(self, *args, **kwargs): self.rows.append(kwargs)
            def tag_configure(self, *args, **kwargs): pass
        def obj(ident, category, diagonal, height):
            return {'id':ident,'image':'a','category_id':category,'diagonal':diagonal,'height':height,'width':2,'area':4}
        objects=[obj('a',1,7,2),obj('b',1,10,2),obj('c',1,10,5),obj('d',2,10,2)]
        ds=SimpleNamespace(images={'a':{'annotations':objects}}, categories={1:'one',2:'two'})
        app=SimpleNamespace(object_tree=Tree(),current_image='a',current=None,ds=ds,gt_sort_column='size',gt_sort_desc=False,edges=[5,9,20],vars={k:Var(v) for k,v in {'category':'1','bin':'9–<20','mode':'Any'}.items()})
        def tags():
            App.populate_objects(app)
            return {r['values'][0]:r['tags'][0] for r in app.object_tree.rows}
        self.assertEqual(tags(),{'a':'other','b':'match','c':'match','d':'other'})
        app.vars['bin'].value=SPECIAL_BIN
        self.assertEqual(tags(),{'a':'other','b':'match','c':'other','d':'other'})
        app.vars['category'].value='All'
        self.assertEqual(tags(),{'a':'other','b':'match','c':'other','d':'match'})
        app.vars['category'].value='2'; app.vars['mode'].value='Without category'
        self.assertEqual(tags(),{'a':'other','b':'other','c':'other','d':'match'})
        self.assertEqual(len(objects_in_images(ds,['a'],'1')),3)
