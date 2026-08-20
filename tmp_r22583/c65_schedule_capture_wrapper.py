from pathlib import Path
p=Path(__file__).with_name('c65_schedule_runner.py')
s=p.read_text(encoding='utf-8')
PINNED_REV='a7ad7e51f7600ec0903f396b700fe1570e1974ec'
old="rev=am['sha']; base_runtime_rev=bm['sha']; card=am.get('cardData') or {}; lic=(card.get('license') or am.get('license') or '').lower(); declared=card.get('base_model')\n    if isinstance(declared,list): declared=declared[0] if len(declared)==1 else declared\n    if lic!=LICENSE: raise SystemExit(31)\n"
new="current_head_seen=am['sha']; rev=PINNED_REV; base_runtime_rev=bm['sha']\n    readme_path=discover(am,'README.md')\n    licdir=escrow/'_license'; licdir.mkdir(exist_ok=True)\n    readme_src=Path(hf_hub_download(repo_id=ADAPTER_REPO,filename=readme_path,revision=rev,local_dir=licdir))\n    readme_text=readme_src.read_text(errors='replace')\n    mlic=re.search(r'(?mi)^license\\s*:\\s*([a-z0-9_.-]+)\\s*$',readme_text)\n    mbase=re.search(r'(?mi)^base_model\\s*:\\s*([^\\s#]+)\\s*$',readme_text)\n    lic=(mlic.group(1).lower() if mlic else '')\n    declared=(mbase.group(1).strip() if mbase else None)\n    readme_sha=sha256_path(readme_src)\n    if lic!=LICENSE: raise SystemExit(f'pinned license mismatch revision={rev} license={lic!r}')\n    if norm_base(declared or EXPECTED_BASE)!=norm_base(EXPECTED_BASE): raise SystemExit(f'pinned base mismatch revision={rev} declared={declared!r}')\n    shutil.rmtree(licdir,ignore_errors=True)\n"
if s.count(old)!=1:
    raise SystemExit(f'expected exactly one original provenance block, found {s.count(old)}')
s=s.replace(old,new)
old_rec="'resolved_revision':rev,'license':lic,'declared_base':declared,"
new_rec="'resolved_revision':rev,'moving_head_seen':current_head_seen,'license':lic,'license_provenance':'same_revision_readme_frontmatter','source_readme_sha256':readme_sha,'declared_base':declared,"
if s.count(old_rec)!=1:
    raise SystemExit(f'expected exactly one receipt provenance field block, found {s.count(old_rec)}')
s=s.replace(old_rec,new_rec)
compile(s,str(p),'exec')
g={'__name__':'__main__','__file__':str(p),'PINNED_REV':PINNED_REV}
exec(compile(s,str(p),'exec'),g,g)
