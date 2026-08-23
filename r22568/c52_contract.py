from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from datetime import date, timedelta

SCHEMA_FIELDS = ["action","date","time","attendees","location","duration","recurrence","notes"]
SEED = 22568
THRESHOLDS = {
    "teacher_ci_low_min": 0.0,
    "teacher_random_margin_min": 0.02,
    "teacher_shuffle_margin_min": 0.02,
    "generation_field_exact_gain_min": 0.10,
    "generation_json_valid_gain_min": 0.05,
    "negative_false_event_increase_max": 0.10,
}

NAMES = ["Alice","Ben","Carla","Dinesh","Eva","Farah","Grace","Hugo"]
LOCS = ["Room A","Room B","HQ lobby","coworking space","library","video call","Studio 4","North cafe"]
ACTIONS = ["meeting","call","lunch","appointment","webinar","training","review","interview"]
DURS = ["30 minutes","45 minutes","1 hour","90 minutes"]
RECUR = ["weekly","daily","monthly",None]
NOTES = [None,"bring the draft","budget review","project kickoff","remote only"]


def canonical_target(action, d, t, attendees=None, location=None, duration=None, recurrence=None, notes=None):
    return {
        "action": action,
        "date": d,
        "time": t,
        "attendees": attendees,
        "location": location,
        "duration": duration,
        "recurrence": recurrence,
        "notes": notes,
    }


def _time12(h, m):
    am = h < 12
    hh = h % 12 or 12
    return f"{hh}:{m:02d} {'AM' if am else 'PM'}"


def _date_ddmmyyyy(d):
    return f"{d.day:02d}/{d.month:02d}/{d.year:04d}"


def _render_absolute(i, d, h, m, action, attendee, location, duration, recurrence, notes, style):
    month = d.strftime('%B')
    if style == 0:
        txt = f"{action.title()} with {attendee} on {d.day} {month} {d.year} at {_time12(h,m)} in {location} for {duration}."
    elif style == 1:
        txt = f"Please schedule a {duration} {action} at {location}, {month} {d.day}, {d.year}, starting {_time12(h,m)}; attendee: {attendee}."
    elif style == 2:
        txt = f"{attendee}: {action} — {d.year}-{d.month:02d}-{d.day:02d} — {_time12(h,m)} — {location} — lasts {duration}."
    else:
        txt = f"Set up {action} for {attendee} at {location} on {d.day:02d}/{d.month:02d}/{d.year} at {_time12(h,m)}, duration {duration}."
    if recurrence:
        txt += f" Repeat {recurrence}."
    if notes:
        txt += f" Note: {notes}."
    return txt


def _render_spoken(i, d, h, m, action, attendee, location, duration, recurrence, notes):
    minute_phrase = "on the hour" if m == 0 else ("half past" if m == 30 else ("quarter past" if m == 15 else f"at {_time12(h,m)}"))
    hour12 = h % 12 or 12
    period = "in the morning" if h < 12 else "in the afternoon"
    if m in (0,15,30):
        when = f"{minute_phrase} {hour12} {period}" if m else f"{hour12} {period}"
    else:
        when = _time12(h,m)
    txt = f"On {d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}, arrange a {action} with {attendee} {when} at {location} for {duration}."
    if recurrence:
        txt += f" Make it {recurrence}."
    if notes:
        txt += f" {notes}."
    return txt


def generate_suite():
    cases=[]
    start=date(2027,1,5)
    splits=[("DISCOVERY",72), ("CONFIRMATION",72), ("TEMPORAL_OOD",64), ("FORMAT_OOD",56), ("RECURRENCE_OOD",56), ("NEGATIVE",64)]
    idx=0
    for split,n in splits:
        for j in range(n):
            cid=f"C52-{idx:03d}"; idx+=1
            if split=="NEGATIVE":
                neg_templates=[
                    "The project deadline is not a calendar event and no meeting should be created.",
                    "Alice mentioned the number 14:30 in a document, but did not schedule anything.",
                    "Do not create an event; this sentence only discusses Room A and next Monday hypothetically.",
                    "There is no appointment, call, meeting, lunch, webinar, training, review, or interview to schedule.",
                ]
                target=canonical_target(None,None,None,None,None,None,None,None)
                cases.append({"id":cid,"split":split,"input":neg_templates[j%len(neg_templates)]+f" Ref {j}.","target":target})
                continue
            d=start+timedelta(days=(j*3 + idx)%330)
            h=8+((j*5+idx)%11); m=[0,15,30,45][(j+idx)%4]
            action=ACTIONS[(j+idx)%len(ACTIONS)]
            attendee=NAMES[(j*2+idx)%len(NAMES)]
            location=LOCS[(j*3+idx)%len(LOCS)]
            duration=DURS[(j+2*idx)%len(DURS)]
            recurrence=RECUR[(j+idx)%len(RECUR)] if split in ("RECURRENCE_OOD","CONFIRMATION") else None
            notes=NOTES[(j+idx)%len(NOTES)] if j%3==0 else None
            style=(j+idx)%4
            if split=="TEMPORAL_OOD":
                text=_render_spoken(idx,d,h,m,action,attendee,location,duration,recurrence,notes)
            elif split=="FORMAT_OOD":
                text=_render_absolute(idx,d,h,m,action,attendee,location,duration,recurrence,notes,2+(style%2))
                text=text.replace(" at "," @ ").replace(" duration ","; duration=")
            else:
                text=_render_absolute(idx,d,h,m,action,attendee,location,duration,recurrence,notes,style)
            target=canonical_target(action,_date_ddmmyyyy(d),_time12(h,m),[attendee],location,duration,recurrence,notes)
            cases.append({"id":cid,"split":split,"input":text,"target":target})
    assert len(cases)==384
    assert len({c['input'] for c in cases})==384
    return cases


def extract_json(text: str):
    text=text.strip()
    starts=[i for i,c in enumerate(text) if c=='{']
    for s in starts:
        depth=0; ins=False; esc=False
        for i in range(s,len(text)):
            ch=text[i]
            if ins:
                if esc: esc=False
                elif ch=='\\': esc=True
                elif ch=='"': ins=False
            else:
                if ch=='"': ins=True
                elif ch=='{': depth+=1
                elif ch=='}':
                    depth-=1
                    if depth==0:
                        try: return json.loads(text[s:i+1])
                        except Exception: break
    return None


def _norm_scalar(v):
    if v is None: return None
    if isinstance(v,str): return re.sub(r"\s+"," ",v.strip()).lower()
    return v


def verifier_a(pred_text, target):
    obj=extract_json(pred_text)
    if not isinstance(obj,dict): return {"json_valid":False,"field_exact":0.0,"exact":False,"false_event":False}
    scores=[]
    for k in SCHEMA_FIELDS:
        a=obj.get(k); b=target.get(k)
        if k=="attendees":
            aa=sorted(_norm_scalar(x) for x in a) if isinstance(a,list) else a
            bb=sorted(_norm_scalar(x) for x in b) if isinstance(b,list) else b
            scores.append(aa==bb)
        else: scores.append(_norm_scalar(a)==_norm_scalar(b))
    exact=all(scores)
    false_event=target.get('action') is None and any(obj.get(k) not in (None,[],"") for k in SCHEMA_FIELDS)
    return {"json_valid":True,"field_exact":sum(scores)/len(scores),"exact":exact,"false_event":bool(false_event)}


def verifier_b(pred_text, target):
    obj=extract_json(pred_text)
    if not isinstance(obj,dict): return {"schema":False,"typed_semantic":0.0,"negative_clean":False}
    schema=set(obj.keys())==set(SCHEMA_FIELDS)
    checks=[]
    # independently validate typed temporal and list semantics
    for k in ("date","time"):
        v=obj.get(k); tv=target.get(k)
        if tv is None: checks.append(v in (None,""))
        elif k=="date": checks.append(isinstance(v,str) and bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}",v)) and v==tv)
        else: checks.append(isinstance(v,str) and bool(re.fullmatch(r"(?:1[0-2]|[1-9]):[0-5]\d (?:AM|PM)",v)) and v==tv)
    a=obj.get('attendees'); ta=target.get('attendees')
    checks.append((a is None and ta is None) or (isinstance(a,list) and isinstance(ta,list) and [str(x).lower() for x in a]==[str(x).lower() for x in ta]))
    for k in ("action","location","duration","recurrence","notes"):
        checks.append(_norm_scalar(obj.get(k))==_norm_scalar(target.get(k)))
    negative_clean = target.get('action') is not None or all(obj.get(k) in (None,[],"") for k in SCHEMA_FIELDS)
    return {"schema":schema,"typed_semantic":sum(checks)/len(checks),"negative_clean":bool(negative_clean)}


def suite_digest(cases=None):
    cases=generate_suite() if cases is None else cases
    raw=json.dumps(cases,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()


def contract_digest():
    raw=json.dumps({"fields":SCHEMA_FIELDS,"thresholds":THRESHOLDS,"suite":suite_digest()},sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()

if __name__=='__main__':
    cases=generate_suite()
    print(json.dumps({"n":len(cases),"suite_sha256":suite_digest(cases),"contract_sha256":contract_digest(),"thresholds":THRESHOLDS},indent=2))
