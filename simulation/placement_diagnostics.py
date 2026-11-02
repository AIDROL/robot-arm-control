"""Read-only simulation diagnostics; no estimates or commands are returned to control."""
import numpy as np
from scipy.spatial.transform import Rotation


def vector_record(value):
    value = np.asarray(value)
    return {'world_xyz_m': value.tolist(), 'xy_m': float(np.linalg.norm(value[:2])),
            'norm_m': float(np.linalg.norm(value))}


def decompose(grasp_offset, nominal_offset, release, final_position,
              target_tcp, target_rotation, target_object):
    """Exact vector identity at release, not a causal allocation of scalar errors."""
    p, R, obj = release['tcp'], release['rotation'], release['object']
    release_offset = R.T @ (obj - p)
    terms = {
        'nominal_geometry': target_tcp + target_rotation @ nominal_offset - target_object,
        'tcp_translation': p - target_tcp,
        'tcp_rotation_lever': (R - target_rotation) @ nominal_offset,
        'grasp_registration': R @ (grasp_offset - nominal_offset),
        'held_relative_change': R @ (release_offset - grasp_offset),
        'post_release_motion': final_position - obj,
    }
    total = final_position - target_object
    return {'terms': {k: vector_record(v) for k, v in terms.items()},
            'final_error': vector_record(total),
            'closure_residual_m': float(np.linalg.norm(sum(terms.values()) - total)),
            'note': 'Signed world vectors add exactly; XY magnitudes do not add. '
                    'Holding change includes transport, compliance and placement contact, not only frictional slip.'}


class PlacementDiagnostics:
    def __init__(self, model, name, target):
        self.body = model.body(name).id
        from .housing_profile import for_model
        self.geometry = for_model(model)
        self.name, self.target = name, np.array(target)
        self.previous = None
        self.endpoints = []
        self.entries = {}
        self.rows = []
        self.last_sample = -1.
        self.held_max_drift = 0.
        self.done_recorded = False

    def observe(self, data, controller):
        phase = controller.phase
        sample = {'time': float(data.time), 'phase': phase.label,
                  'tcp': data.site_xpos[controller.site].copy(),
                  'rotation': data.site_xmat[controller.site].reshape(3, 3).copy(),
                  'object': data.xpos[self.body].copy(),
                  'object_rotation': data.xmat[self.body].reshape(3, 3).copy(),
                  'target': phase.position.copy(), 'target_rotation': phase.orientation.as_matrix()}
        changed = self.previous is None or self.previous['phase'] != phase.label
        if changed:
            if self.previous is not None:
                self._endpoint(self.previous, sample)
            self.entries[phase.label] = sample
        if controller.done and not self.done_recorded:
            self._endpoint(sample, sample)
            self.done_recorded = True
        baseline = self.entries.get(f'{self.name}: lift')
        if baseline is not None and f'{self.name}: release' not in self.entries:
            C0 = baseline['rotation'].T @ (baseline['object'] - baseline['tcp'])
            C = sample['rotation'].T @ (sample['object'] - sample['tcp'])
            self.held_max_drift = max(self.held_max_drift, float(np.linalg.norm(C-C0)))
        if changed or data.time - self.last_sample >= .1:
            self.rows.append({k: v.tolist() if isinstance(v, np.ndarray) else v
                              for k, v in sample.items()})
            self.last_sample = float(data.time)
        self.previous = sample

    def _endpoint(self, previous, sample):
        delta = sample['tcp'] - previous['target']
        angle = Rotation.from_matrix(previous['target_rotation'].T @ sample['rotation']).magnitude()
        self.endpoints.append({'phase': previous['phase'], 'time': sample['time'],
                               'tcp_position_error': vector_record(delta),
                               'tcp_rotation_error_rad': float(angle)})

    def result(self, data, controller):
        result = {'frame_id': 'world', 'units': 'metres/radians',
                  'phase_endpoints': self.endpoints, 'holding_max_relative_drift_m': self.held_max_drift,
                  'complete': False}
        grasp = self.entries.get(f'{self.name}: lift')
        release = self.entries.get(f'{self.name}: release')
        if grasp is None or release is None or not controller.done:
            result['reason'] = 'INCOMPLETE_GRASP_OR_RELEASE'
            return result
        approach = next(p for p in controller.phases if p.label == f'{self.name}: approach')
        lower = next(p for p in controller.phases if p.label == f'{self.name}: lower')
        # Configured nominal grasp is 10 mm above the housing centre in world Z.
        nominal = approach.orientation.inv().apply([0., 0., -self.geometry.grasp_z])
        grasp_offset = grasp['rotation'].T @ (grasp['object'] - grasp['tcp'])
        result.update(decompose(grasp_offset, nominal, release, data.xpos[self.body].copy(),
                                lower.position, lower.orientation.as_matrix(), self.target))
        result['complete'] = True
        result['release_time'] = release['time']
        result['grasp_time'] = grasp['time']
        retreat = self.entries.get(f'{self.name}: retreat')
        if retreat is not None:
            result['release_motion_detail'] = {
                'during_gripper_opening': vector_record(retreat['object'] - release['object']),
                'after_opening': vector_record(data.xpos[self.body] - retreat['object'])}
        return result


def markdown_report(result):
    lines = ['# 放置误差诊断', '', '以下为仿真真值诊断，不参与控制。', '',
             '| 阶段结束 | TCP XY 误差 mm | TCP 三维误差 mm |', '| --- | ---: | ---: |']
    for p in result['phase_endpoints']:
        e = p['tcp_position_error']
        lines.append(f"| {p['phase']} | {e['xy_m']*1000:.4f} | {e['norm_m']*1000:.4f} |")
    if result['complete']:
        lines += ['', '| 误差项 | X mm | Y mm | XY 模长 mm |', '| --- | ---: | ---: | ---: |']
        for k, e in {**result['terms'], 'final_error': result['final_error']}.items():
            x, y, _ = np.array(e['world_xyz_m'])*1000
            lines.append(f"| {k} | {x:.4f} | {y:.4f} | {e['xy_m']*1000:.4f} |")
        lines += ['', '各项是可带正负号的世界坐标向量，其向量和等于最终误差；模长不能直接相加。',
                  '夹持相对变化包括运输、夹爪柔顺和下降接触，不能直接认定都是摩擦滑移。']
        if 'release_motion_detail' in result:
            lines += ['', '| 释放细分 | XY 位移 mm |', '| --- | ---: |']
            for key, value in result['release_motion_detail'].items():
                lines.append(f"| {key} | {value['xy_m']*1000:.6f} |")
    return '\n'.join(lines)+'\n'
