import unittest,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from r128.compat import bridge_layernorm_aliases
class TestLegacy(unittest.TestCase):
 def test_gamma_beta_bridge(self):
  w=torch.randn(3);b=torch.randn(3)
  s={'bert.LayerNorm.gamma':w,'bert.LayerNorm.beta':b,'bert.weight':w}
  y,c=bridge_layernorm_aliases(s,['bert.LayerNorm.weight','bert.LayerNorm.bias','bert.weight'])
  self.assertEqual(len(c),2)
  self.assertTrue(torch.equal(y['bert.LayerNorm.weight'],w))
  self.assertTrue(torch.equal(y['bert.LayerNorm.bias'],b))
 def test_strict_other_key(self):
  s={'evil.gamma':torch.ones(1)}
  y,c=bridge_layernorm_aliases(s,['evil.weight'])
  self.assertEqual(len(c),0);self.assertIn('evil.gamma',y)
 def test_no_override(self):
  s={'bert.LayerNorm.weight':torch.ones(1),'bert.LayerNorm.gamma':torch.ones(1)}
  with self.assertRaises(ValueError):bridge_layernorm_aliases(s,['bert.LayerNorm.weight'])
 def test_alias_collision(self):
  s={'bert.LayerNorm.gamma':torch.ones(1),'bert.LayerNorm.weight':torch.ones(1)}
  with self.assertRaises(ValueError):bridge_layernorm_aliases(s,['bert.LayerNorm.weight'])
if __name__=='__main__':unittest.main()
