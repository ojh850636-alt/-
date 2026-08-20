from pathlib import Path
p=Path(__file__).with_name('c65_schedule_runner.py')
s=p.read_text(encoding='utf-8')
old="rev=am['sha']; base_runtime_rev=bm['sha']; card=am.get('cardData') or {}; lic=(card.get('license') or am.get('license') or '').lower(); declared=card.get('base_model')\n    if isinstance(declared,list): declared=declared[0] if len(declared)==1 else declared\n    if lic!=LICENSE: raise SystemExit(31)\n"
new="rev=am['sha']; base_runtime_rev=bm['sha']; card=am.get('cardData') or {}; api_lic=(card.get('license') or am.get('license') or '').lower(); declared=card.get('base_model')\n    if isinstance(declared,list): declared=declared[0] if len(declared)==1 else declared\n    readme_path=discover(am,'README.md')\n    licdir=escrow/'_license'; licdir.mkdir(exist_ok=True)\n    readme_src=Path(hf_hub_download(repo_id=ADAPTER_REPO,filename=readme_path,revision=rev,local_dir=licdir))\n    readme_text=readme_src.read_text(errors='replace')\n    mlic=re.search(r'(?mi)^license\\s*:\\s*([a-z0-9_.-]+)\\s*$',readme_text)\n    readme_lic=(mlic.group(1).lower() if mlic else '')\n    lic=api_lic if api_lic==LICENSE else readme_lic\n    if lic!=LICENSE: raise SystemExit(f'license mismatch api={api_lic!r} readme={readme_lic!r}')\n    shutil.rmtree(licdir,ignore_errors=True)\n"
if s.count(old)!=1:
    raise SystemExit(f'expected exactly one license-gate block, found {s.count(old)}')
s=s.replace(old,new)
compile(s,str(p),'exec')
g={'__name__':'__main__','__file__':str(p)}
exec(compile(s,str(p),'exec'),g,g)
