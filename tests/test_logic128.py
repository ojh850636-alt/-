import unittest, sys, random, json, zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r128.logic import paired,paired_bootstrap,gate,digest

class TestPaired(unittest.TestCase):
 def test_rescue_p(self):
  a=[1]*12;b=[0]*12;y=[0]*12
  x=paired(a,b,y);self.assertEqual((x['rescue'],x['harm']), (12,0));self.assertAlmostEqual(x['one_sided_p'],1/4096)
 def test_harm(self):
  a=[0]*8;b=[1]*8;y=[0]*8
  x=paired(a,b,y);self.assertEqual(x['delta'],-1)
  self.assertEqual(x['one_sided_p'],1)
 def test_neutral(self):
  a=[0,1,0,1];self.assertEqual(paired(a,a,a)['one_sided_p'],1)
 def test_mixed(self):
  self.assertEqual(paired([1,0,0,1],[0,1,1,1],[0,0,1,1])['delta'],0.25)
 def test_invalid_len(self):
  with self.assertRaises(ValueError):paired([0],[0],[0,1])
 def test_invalid_label(self):
  with self.assertRaises(ValueError):paired([0],[1],[7])
 def test_empty(self):
  with self.assertRaises(ValueError):paired([],[],[])
 def test_bootstrap_positive(self):
  a=[1]*30;b=[0]*30;y=[0]*30
  self.assertEqual(paired_bootstrap(a,b,y,draws=50),[1,1])
 def test_bootstrap_repeatable(self):
  a=[1,0]*10;b=[0]*20;y=[0]*20
  self.assertEqual(paired_bootstrap(a,b,y,draws=40),paired_bootstrap(a,b,y,draws=40))
 def test_digest(self):self.assertEqual(digest({'b':1,'a':2}),digest({'a':2,'b':1}))
 def test_gate_red_without_controls(self):
  p=json.loads((Path(__file__).parents[1]/'data/POLICY128.json').read_text())
  def s(a,b,rand,shuffle):return {'by_condition':{'FULL':a,'RANDOM_SAME_HEAD':rand,'SHUFFLED_SAME_HEAD':shuffle},'full_vs_head':{'delta':b,'one_sided_p':0.0001}}
  out=gate({'ALL':s(.8,.2,.9,.9),'CONFIRMATION':s(.8,.2,.3,.3),'FAMILY_OOD':s(.8,.1,.3,.3)},p)
  self.assertEqual(out['behavior_positive_screen'],'RED')
 def test_gate_green_controlled(self):
  p=json.loads((Path(__file__).parents[1]/'data/POLICY128.json').read_text())
  def s(a,b,rand,shuffle):return {'by_condition':{'FULL':a,'RANDOM_SAME_HEAD':rand,'SHUFFLED_SAME_HEAD':shuffle},'full_vs_head':{'delta':b,'one_sided_p':0.0001}}
  out=gate({'ALL':s(.8,.2,.7,.7),'CONFIRMATION':s(.8,.2,.3,.3),'FAMILY_OOD':s(.8,.1,.3,.3)},p)
  self.assertEqual(out['behavior_positive_screen'],'GREEN')
  self.assertFalse(out['causal_claim'])
if __name__=='__main__':unittest.main()
