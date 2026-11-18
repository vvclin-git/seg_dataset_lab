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
