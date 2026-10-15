"""A new local closed-loop flight, never a replay or remote computation."""
from __future__ import annotations
import asyncio
import hashlib
import json
import math
import sys
import time
import numpy as np

sys.path[:0] = ['/gift/physics','/gift/policy']
from browser_physics import Flight, CHECKPOINT
from browser_policy import BrowserPolicy


def progress(**record):
    from js import gift_progress
    gift_progress(json.dumps(record,ensure_ascii=False,allow_nan=False))


def cte(position, reference):
    """Distance to the drawn reference polyline, with bounded temporary memory."""
    best = float('inf')
    point = np.asarray(position,np.float64)
    for offset in range(0,len(reference)-1,8192):
        a=np.asarray(reference[offset:offset+8192],np.float64)
        b=np.asarray(reference[offset+1:offset+8193],np.float64)
        a=a[:len(b)];delta=b-a
        fraction=np.clip(np.sum((point-a)*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1.e-20),0.,1.)
        error=point-(a+fraction[:,None]*delta)
        best=min(best,float(np.min(np.sum(error*error,axis=1))))
    return math.sqrt(best)


def acmi(trace, name):
    lines=['FileType=text/acmi/tacview','FileVersion=2.2',
        '0,ReferenceTime=2026-10-15T00:00:00Z','0,Title='+name.replace(',',' '),
        '0,Author=GIFT local A4 flight','#0.00']
    for i,p in enumerate(trace['reference']):
        lines.append(f'R{i:X},T={p[1]/111320.:.8f}|{p[0]/111320.:.8f}|{p[2]:.3f},Type=Navaid+Static+Waypoint,Color=Green'+(f',Next=R{i+1:X}' if i+1<len(trace['reference']) else ''))
    for t,p,h in zip(trace['time'],trace['position'],trace['hprv']):
        lines.extend([f'#{t:.3f}',f'100,T={p[1]/111320.:.8f}|{p[0]/111320.:.8f}|{p[2]:.3f}|{math.degrees(h[2]):.5f}|{math.degrees(h[1]):.5f}|{math.degrees(h[0]):.5f},Type=Air+FixedWing,Name=A4,Color=Cyan,CallSign=Local frozen baseline'])
    return '\n'.join(lines)+'\n'


async def run_json(raw):
    import js
    request=json.loads(raw);flight_id=request.pop('id');audit_inputs=bool(request.pop('audit',False));started=time.perf_counter()
    progress(status='compiling',message='在本机将完整三维路线编译为原 baseline 目标流')
    flight=Flight(request);geometry_seconds=time.perf_counter()-started
    progress(status='loading',message='载入完整冻结 baseline，保留全部策略状态',compilation=flight.compilation)
    policy=BrowserPolicy('/gift/policy/policy_manifest.json',record=False)
    samples=[flight.sample()];actions=[];inputs=[];error=None;flight_started=time.perf_counter()
    while not flight.done:
        if bool(js.gift_cancelled):
            flight.cancel();break
        try:
            observation,context=flight.observe(),flight.context()
            if audit_inputs:inputs.append(dict(observation=observation.tolist(),context=context.tolist(),carry=policy.carry.tolist(),reset=policy.control==0))
            value=policy.step(observation,context=context)
            action=np.asarray(value[1],np.int32).reshape(5)
            sample=flight.advance(action)
            if not all(math.isfinite(float(x)) for x in sample['position']+sample['hprv']):
                error='动力学产生非有限状态，保留此前真实计算的航迹';break
            samples.append(sample);actions.append(action.tolist())
        except Exception as exc:
            error=f'本机飞行计算在第 {len(actions)} 步停止：{exc}'
            break
        if len(actions)%5==0:
            progress(status='flying',message='本地 baseline 推理与真实 A4 动力学计算',
                controls=len(actions),seconds=sample['time'],progress=sample['progress'])
            await asyncio.sleep(0)
    reference=np.asarray(flight.reference,np.float64)
    trace={key:[s[key] for s in samples] for key in ['time','position','hprv','alpha','beta','g']}
    trace.update(reference=reference.tolist(),cte=[cte(s['position'],reference) for s in samples],
        synthetic_flight_samples=False,attitude_complete=True,
        source_sha256=hashlib.sha256(np.asarray([s['position'] for s in samples],'<f4').tobytes()).hexdigest())
    note=error or {'route_end':'已飞至路线末段','no_geometric_progress':'飞机持续未取得路线进展，已保留实际航迹',
        'physical_termination':'触发真实动力学终止条件，已保留实际航迹','nonfinite_physics':'动力学出现非有限状态，已保留此前航迹',
        'cancelled':'已取消本机试飞，保留已飞航迹'}.get(flight.reason,'本机实际航迹已生成')
    distances=np.asarray(trace['cte'],np.float64)
    meta=dict(checkpoint=CHECKPOINT,flown=len(actions)>0,name=request.get('name','自定义三维路线'),note=note,
        progress=float(flight.progress.fraction),status='recorded',seconds=samples[-1]['time'],
        cteP90=float(np.percentile(distances,90)),cteMax=float(np.max(distances)),
        endpoint=float(np.linalg.norm(np.asarray(samples[-1]['position'])-reference[-1])),
        safety=flight.reason not in ('physical_termination','nonfinite_physics'),spin=None,reasons=[],
        execution='browser-local',cte_definition='distance_to_reference_polyline',server_calls=0,bitwise_gpu_equivalence_claim=False)
    result=dict(id=flight_id,status='cancelled' if flight.reason=='cancelled' else 'completed',
        meta=meta,trace=trace,actions=actions,compilation=flight.compilation,message=note,
        reason=flight.reason,policy=policy.report(),physics=flight.identity(),
        timings=dict(geometry_seconds=geometry_seconds,flight_seconds=time.perf_counter()-flight_started),
        teacher_calls=0,optimizer_updates=0,checkpoint_switches=0,wall_seconds=time.perf_counter()-started)
    result['acmi']=acmi(trace,meta['name'])
    if audit_inputs:result['policy_inputs']=inputs
    return json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))
