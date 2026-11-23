import unittest,sys,hashlib,json,zipfile,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r128.worker import verify_suite,SUITE_SHA
from tools.emit_suite128 import emit

class TestContract(unittest.TestCase):
 def test_parent_suite_sha(self):
  with zipfile.ZipFile(Path(__file__).parents[1]/'P127.zip') as z:
   b=z.read('data/SCIENCE_SUITE127.jsonl')
  self.assertEqual(hashlib.sha256(b).hexdigest(),SUITE_SHA)
 def test_suite_validator_from_outside(self):
  with tempfile.TemporaryDirectory() as t:
   p=emit(Path(t)/'suite.jsonl')
   self.assertEqual(len(verify_suite(p)),240)
 def test_authorized_source_not_closed(self):
  root=Path(__file__).parents[1]
  d=json.loads((root/'data/SOURCE128.json').read_text())
  self.assertEqual(d['winner'],'RTE')
  self.assertNotIn('rte/seed0',d['prior_closed_sources'])
 def test_gate_precommit_contains_both_lanes(self):
  d=json.loads((Path(__file__).parents[1]/'data/POLICY128.json').read_text())
  self.assertIn('failure_gate',d);self.assertIn('behavior_gate',d)
 def test_preseal_file_digest(self):
  r=Path(__file__).parents[1];p=json.loads((r/'data/PRESEAL128.json').read_text())
  for path,want in p['files'].items():
   got=hashlib.sha256((r/path).read_bytes()).hexdigest()
   if got!=want and path=='r128/worker.py':
    amend=json.loads((r/'data/AMEND128.json').read_text())
    self.assertEqual(want,amend['old_worker_sha256'])
    self.assertEqual(got,amend['new_worker_sha256'])
    self.assertEqual(hashlib.sha256((r/'r128/compat.py').read_bytes()).hexdigest(),amend['compat_code_sha256'])
   else:self.assertEqual(got,want,path)
 def test_raw_free_claim(self):
  state_path=Path(__file__).parents[1]/'data/CURRENT128.json'
  if not state_path.exists():self.skipTest('CURRENT128 written only after actual report')
  s=json.loads(state_path.read_text());self.assertEqual(s['new_external_E4_E5_count'],0)
if __name__=='__main__':unittest.main()
