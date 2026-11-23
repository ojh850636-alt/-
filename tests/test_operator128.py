import unittest,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from r128.adapter import parse_lora_tensors,stat_operator,Controller,randomized_b,selection

class TNet(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=torch.nn.Module()
  self.encoder.layer=torch.nn.ModuleList([torch.nn.Module()])
  self.encoder.layer[0].attention=torch.nn.Module();self.encoder.layer[0].attention.query=torch.nn.Linear(3,3)
  self.classifier=torch.nn.Linear(3,2)
 def forward(self,x):return self.classifier(self.encoder.layer[0].attention.query(x))

def fake():
 torch.manual_seed(34)
 a=torch.randn(2,3);b=torch.randn(3,2)*.5
 head_w=torch.randn(2,3);head_b=torch.randn(2)
 sd={'base_model.model.encoder.layer.0.attention.query.lora_A.default.weight':a,
     'base_model.model.encoder.layer.0.attention.query.lora_B.default.weight':b,
     'base_model.model.classifier.modules_to_save.default.weight':head_w,
     'base_model.model.classifier.modules_to_save.default.bias':head_b}
 return sd

class TestOperator(unittest.TestCase):
 def test_parse(self):
  f,h=parse_lora_tensors(fake(),2)
  self.assertIn('encoder.layer.0.attention.query',f)
  self.assertEqual(len(h),2)
 def test_static_dense_matches(self):
  f,h=parse_lora_tensors(fake(),2)
  x=stat_operator(f,alpha=4)[0]
  a=f[x['site']]['A'];b=f[x['site']]['B']
  self.assertAlmostEqual(x['operator_norm_sq'],float(torch.linalg.norm(2*b@a)**2),places=4)
 def test_model_conditional_head_orthogonal(self):
  f,h=parse_lora_tensors(fake(),2)
  net=TNet();x=torch.tensor([[1.,2.,3.],[.1,.2,.3]])
  ctrl=Controller(net,f,h,alpha=4)
  try:
   ctrl.condition('BASE');base=net(x).detach().clone()
   ctrl.condition('HEAD_ONLY');head=net(x).detach().clone()
   ctrl.condition('LORA_ONLY');lora=net(x).detach().clone()
   ctrl.condition('FULL');full=net(x).detach().clone()
   self.assertFalse(torch.allclose(base,head))
   self.assertFalse(torch.allclose(base,lora))
   self.assertFalse(torch.allclose(head,full))
   ctrl.condition('FULL',dose=0);self.assertTrue(torch.allclose(net(x),head))
   ctrl.condition('LORA_ONLY',dose=0);self.assertTrue(torch.allclose(net(x),base))
  finally:ctrl.close()
 def test_rank_mismatch_fail(self):
  sd=fake();sd['base_model.model.encoder.layer.0.attention.query.lora_A.default.weight']=torch.randn(4,3)
  with self.assertRaises(ValueError):parse_lora_tensors(sd,2)
 def test_missing_head_fail(self):
  sd=fake();del sd['base_model.model.classifier.modules_to_save.default.bias']
  with self.assertRaises(ValueError):parse_lora_tensors(sd,2)
 def test_missing_B_fail(self):
  sd=fake();del sd['base_model.model.encoder.layer.0.attention.query.lora_B.default.weight']
  with self.assertRaises(ValueError):parse_lora_tensors(sd,2)
 def test_nan_fail(self):
  sd=fake();key=list(sd)[0];sd[key][0,0]=float('nan')
  with self.assertRaises(ValueError):parse_lora_tensors(sd,2)
 def test_unknown_param_fail(self):
  sd=fake();sd['unsafe.extra']=torch.zeros(1)
  with self.assertRaises(ValueError):parse_lora_tensors(sd,2)
 def test_random_B_norm(self):
  f,h=parse_lora_tensors(fake(),2)
  k=list(f)[0];r=randomized_b(f,'RANDOM',seed=5)
  self.assertAlmostEqual(float(torch.linalg.norm(r[k][1])),float(torch.linalg.norm(f[k]['B'])),places=5)
 def test_shuffled_control_repeatable(self):
  f,h=parse_lora_tensors(fake(),2)
  k=list(f)[0];r=randomized_b(f,'SHUFFLED',seed=12)
  s=randomized_b(f,'SHUFFLED',seed=12)
  self.assertTrue(torch.equal(r[k][1],s[k][1]))
 def test_gauge_scale_stability(self):
  f,h=parse_lora_tensors(fake(),2)
  k=list(f)[0];q=stat_operator(f,alpha=4)[0]['operator_norm_sq']
  f[k]['A']=f[k]['A']*7;f[k]['B']=f[k]['B']/7
  self.assertAlmostEqual(stat_operator(f,alpha=4)[0]['operator_norm_sq'],q,places=4)
 def test_group(self):
  self.assertTrue(selection('bert.encoder.layer.11.attention.output.dense','ATTENTION'))
  self.assertTrue(selection('bert.encoder.layer.11.output.dense','MLP'))
  self.assertTrue(selection('bert.encoder.layer.11.output.dense','LAYER_BAND_LATE'))
  self.assertFalse(selection('bert.encoder.layer.3.output.dense','LAYER_BAND_LATE'))
 def test_negative_scope(self):
  with self.assertRaises(ValueError):selection('something','BAD_GROUP')
 def test_suppression_affects_output(self):
  f,h=parse_lora_tensors(fake(),2)
  net=TNet();ctrl=Controller(net,f,h,alpha=4);x=torch.ones(1,3)
  try:
   ctrl.condition('FULL');t=net(x).detach().clone()
   ctrl.suppressed=set(f)
   s=net(x).detach().clone()
   ctrl.condition('HEAD_ONLY');h_=net(x).detach().clone()
   self.assertTrue(torch.allclose(s,h_,atol=1e-6))
   self.assertFalse(torch.allclose(t,s,atol=1e-6))
  finally:ctrl.close()
 def test_head_state_restored(self):
  f,h=parse_lora_tensors(fake(),2)
  net=TNet();pre=net.classifier.weight.detach().clone();ctrl=Controller(net,f,h,alpha=4)
  ctrl.condition('FULL');ctrl.close();self.assertTrue(torch.equal(net.classifier.weight.detach(),pre))
if __name__=='__main__':unittest.main()
