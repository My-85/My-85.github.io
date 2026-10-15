"""Geometry-only demo compiler; diagnostics are advisory, not certificates.

Keep the complete path and adapt HPRV commands to the baseline ABI in one
forward pass. No policy, aircraft rollout, inverse search or optimizer input.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA = 'planax.a4.demo_open_geometry_stream.v1'
DT = .2
G = 9.80665


def prepare(request):
    if not isinstance(request, dict) or set(request)-{'positions', 'name', 'control_points', 'control_segments'}:
        raise ValueError('只接受路线坐标、名称和控制点对应关系')
    try:
        p = np.asarray(request.get('positions'), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('路线坐标必须是有限数值 [北向, 东向, 高度]') from exc
    if p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('需要至少两个有限三维坐标点')
    controls = np.asarray(request.get('control_points', p), dtype=float)
    if controls.ndim != 2 or controls.shape[1:] != (3,) or len(controls) < 2 or not np.isfinite(controls).all():
        raise ValueError('控制点必须是有限三维坐标')
    segments = request.get('control_segments')
    if segments is not None and (len(segments) != len(p) or any(type(i) is not int or not 0 <= i < len(controls)-1 for i in segments)):
        raise ValueError('采样点与控制点对应关系不合法')
    # Remove only consecutive coincident samples; later returns are preserved.
    keep = np.r_[True, np.any(np.diff(p, axis=0) != 0., axis=1)]
    source = np.flatnonzero(keep); p = p[keep]
    if len(p) < 2:
        raise ValueError('路线没有长度，请移动或添加一个点')
    if len(p) == 2:
        p = np.vstack((p[0], (p[0]+p[1])/2., p[1]))
        source = np.asarray([source[0], source[0], source[-1]])
    phase = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if not np.isfinite(phase).all() or np.any(np.diff(phase) <= 0.):
        raise ValueError('坐标超出数值精度，请检查坐标格式')
    initial = p[1]-p[0]
    heading = float(np.degrees(np.arctan2(initial[1], initial[0])))
    return dict(positions=p, phase=phase, heading=heading, controls=controls,
        control_segments=None if segments is None else np.asarray(segments)[source],
        source_indices=source, sha256=hashlib.sha256(p.astype('<f8').tobytes()).hexdigest(),
        name=str(request.get('name', '自定义三维路线'))[:60])


@dataclass(frozen=True)
class DemoStream:
    targets: np.ndarray
    raw_targets: np.ndarray
    phase_s_m: np.ndarray
    positions_neu_m: np.ndarray
    geometry_sha256: str
    stream_sha256: str
    case: str = 'custom_demo'
    control_dt_s: float = DT
    algorithm_version: str = SCHEMA
    source_kind: str = 'uncertified_demo_geometry'

    @property
    def total_length_m(self):
        return float(self.phase_s_m[-1])


def synthesize(g):
    p, s = g['positions'], g['phase']
    tangent = np.gradient(p, s, axis=0, edge_order=1)
    norm = np.linalg.norm(tangent, axis=1)
    # At an exact reversal the centred derivative is ambiguous. Use the next
    # segment, retaining the reversal and recording any command adaptation.
    edges = np.diff(p, axis=0); outgoing = np.vstack((edges, edges[-1]))
    tangent[norm < 1.e-12] = outgoing[norm < 1.e-12]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    curvature = np.gradient(tangent, s, axis=0, edge_order=1)
    horizontal = np.hypot(tangent[:, 0], tangent[:, 1])
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    for i in np.flatnonzero(horizontal < 1.e-9):
        heading[i] = heading[i-1] if i else np.radians(g['heading'])
    gamma = np.arctan2(tangent[:, 2], horizontal)
    speed = np.clip(np.sqrt(G*2.4/np.maximum(np.linalg.norm(curvature, axis=1), 1.e-12)), 170., 250.)
    force = speed[:, None]**2*curvature + np.array([0., 0., G])
    sideways = np.column_stack((-np.sin(heading), np.cos(heading), np.zeros(len(s))))
    up = np.column_stack((-np.sin(gamma)*np.cos(heading), -np.sin(gamma)*np.sin(heading), np.cos(gamma)))
    roll = np.arctan2(np.sum(force*sideways, axis=1), np.sum(force*up, axis=1))
    load = np.linalg.norm(force-tangent*np.sum(force*tangent, axis=1)[:, None], axis=1)/G
    pitch = np.clip(gamma+np.radians(np.clip(.6*load, 0., 8.)), -np.pi/2., np.pi/2.)
    raw = np.column_stack((heading, pitch, roll, speed)); targets = raw.copy()
    targets[0] = [np.radians(g['heading']), np.radians(.6), 0., 250.]
    limits = np.radians([30., 16., 40.])*DT
    for i in range(1, len(s)):
        delta = raw[i, :3]-targets[i-1, :3]
        delta[[0, 2]] = np.angle(np.exp(1j*delta[[0, 2]]))
        # Coordinate variation also bounds physical preview rotation.
        scale = max(1., float(np.max(np.abs(delta)/limits)), float(np.sum(np.abs(delta))/np.radians(45.*DT)))
        targets[i, :3] = targets[i-1, :3]+delta/scale
        targets[i, 3] = targets[i-1, 3]+np.clip(raw[i, 3]-targets[i-1, 3], -15.*DT, 15.*DT)
    targets[:, [0, 2]] = np.angle(np.exp(1j*targets[:, [0, 2]]))
    raw[:, [0, 2]] = np.angle(np.exp(1j*raw[:, [0, 2]]))
    digest = stream_digest(targets, raw, p, s)
    return DemoStream(targets, raw, s, p, g['sha256'], digest), load, gamma


def stream_digest(targets, raw, positions, phase):
    h = hashlib.sha256(SCHEMA.encode())
    for a in (targets, raw, positions, phase):
        a = np.asarray(a, '<f8'); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def diagnose(g, stream, load, gamma):
    s, p = g['phase'], g['positions']
    angular = np.angle(np.exp(1j*(stream.targets[:, :3]-stream.raw_targets[:, :3])))
    limited = np.any(np.abs(angular) > np.radians(.2), axis=1) | (np.abs(stream.targets[:, 3]-stream.raw_targets[:, 3]) > .2)
    definitions = [
        ('load', load > 7.5, '急弯或高度突变', '几何估算载荷超过 7.5 g；目标已兼容处理，可拉开附近控制点以减小偏差。'),
        ('slope', np.abs(gamma) > np.radians(35.), '陡坡或垂直段', '坡度超过 35°；仍会试飞，可拉长此段或降低相邻点高度差。'),
        ('altitude', (p[:, 2] < 2850.) | (p[:, 2] > 8500.), '超出原高度检查范围', '此段超出原高度检查范围；现在允许编译和试飞，实际表现以飞行记录为准。'),
        ('compatible_targets', limited, '目标变化已兼容处理', '此段姿态或速度变化较快，已生成 baseline 可读取的目标；原始路线坐标保留。'),
    ]
    issues = []
    for code, mask, title, message in definitions:
        indices = np.flatnonzero(mask)
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
            if not len(run): continue
            a, b = max(0, int(run[0])-1), min(len(s)-1, int(run[-1])+1)
            worst = int(run[np.argmax(load[run])]) if code == 'load' else int(run[len(run)//2])
            if g['control_segments'] is not None:
                segments = sorted(set(g['control_segments'][a:b+1].tolist()))
                near = sorted(set(segments+[i+1 for i in segments]))
            else:
                near = np.argsort(np.linalg.norm(g['controls']-p[worst], axis=1))[:2].tolist()
            issues.append(dict(code=code, title=title, message=message, severity='warning',
                sample_start=int(g['source_indices'][a]), sample_end=int(g['source_indices'][b]),
                station_start_m=float(s[a]), station_end_m=float(s[b]), station_m=float(s[worst]),
                position=p[worst].tolist(), point_indices=near, estimated_load_g=float(load[worst]),
                slope_deg=float(np.degrees(gamma[worst]))))
    return dict(schema=SCHEMA+'.diagnostics', status='compiled', certified=False,
        geometry_sha256=g['sha256'], length_m=float(s[-1]), target_rows=len(s), issues=issues,
        compatibility_adjusted_rows=int(np.count_nonzero(limited)),
        message=f'已编译完整路线 {s[-1]/1000:.2f} km；'+(f'{len(issues)} 处提示，点击可定位，继续 baseline 试飞。' if issues else '继续 baseline 试飞。'))


def compile_request(request, output):
    g = prepare(request); folder = Path(output); folder.mkdir()
    stream, load, gamma = synthesize(g); diagnostics = diagnose(g, stream, load, gamma)
    compiled = folder/'compiled'; compiled.mkdir()
    np.savez_compressed(compiled/'arrays.npz', targets=stream.targets, raw_targets=stream.raw_targets,
        phase_s_m=stream.phase_s_m, positions_neu_m=stream.positions_neu_m)
    manifest = dict(schema=SCHEMA, geometry_sha256=g['sha256'], stream_sha256=stream.stream_sha256,
        arrays_sha256=hashlib.sha256((compiled/'arrays.npz').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), certified=False)
    (compiled/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    end = g['positions'][-1]-g['positions'][-2]
    record = dict(index=0, case_id='interactive_'+g['sha256'][:16], split='development',
        cohort='user_geometry', entry_heading_deg=g['heading'], name_zh=g['name'],
        start_position_neu_m=g['positions'][0].tolist(),
        requested_terminal_heading_deg=float(np.degrees(np.arctan2(end[1], end[0]))))
    entry = dict(record=record, compiled=dict(folder='compiled', stream_sha256=stream.stream_sha256,
        manifest_sha256=hashlib.sha256((compiled/'manifest.json').read_bytes()).hexdigest()),
        admitted=True, all_admissions_passed=False, execution_error=False,
        status='demo_compiled', demo_stream=True, compilation=diagnostics)
    (folder/'entry.json').write_text(json.dumps(entry, ensure_ascii=False, allow_nan=False))
    (folder/'diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, allow_nan=False))
    return entry


def load_stream(root, expected_sha):
    root = Path(root); manifest_path = root/'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('自定义编译清单校验失败')
    m = json.loads(manifest_path.read_text())
    if m['schema'] != SCHEMA or hashlib.sha256((root/'arrays.npz').read_bytes()).hexdigest() != m['arrays_sha256']:
        raise ValueError('自定义目标流数据校验失败')
    with np.load(root/'arrays.npz', allow_pickle=False) as z: a = {k:z[k] for k in z.files}
    n = len(a['targets'])
    if a['targets'].shape != (n, 4) or a['raw_targets'].shape != (n, 4) or a['positions_neu_m'].shape != (n, 3) or a['phase_s_m'].shape != (n,) or n < 3:
        raise ValueError('自定义目标流形状不匹配')
    if not all(np.isfinite(v).all() for v in a.values()) or np.any(np.diff(a['phase_s_m']) <= 0.) or np.any(a['targets'][:, 3] <= 0.):
        raise ValueError('自定义目标流数值不合法')
    digest = stream_digest(a['targets'], a['raw_targets'], a['positions_neu_m'], a['phase_s_m'])
    if digest != m['stream_sha256']: raise ValueError('自定义目标流身份不匹配')
    return DemoStream(**a, geometry_sha256=m['geometry_sha256'], stream_sha256=digest)
