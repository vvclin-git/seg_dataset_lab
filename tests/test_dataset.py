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
