"""Three-mount paraxial scalar-wave check, independent of robot control."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_actual(layout_path, checkpoint, output):
    """Rebuild the exact assembly model and read sites from its saved state."""
    import mujoco
    from . import line_assembly as scene
    from .optical_layout import load
    from .housing_profile import STL
    plan = load(layout_path)
    saved = np.load(checkpoint, allow_pickle=False)
    expected = STL.layout_id+':'+plan.sha256
    if str(saved['housing_geometry']) != STL.id or str(saved['target_layout']) != expected:
        raise ValueError('Checkpoint geometry/layout mismatch')
    model = scene.make_scene(seed=int(saved['seed']), include_fat=True, vision=True,
                             target_xs=plan.target_xs, component_names=plan.names,
                             layout_sha256=plan.sha256)
    data = mujoco.MjData(model)
    if saved['state'].shape != (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),):
        raise ValueError('Invalid state shape')
    mujoco.mj_setState(model, data, saved['state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    doc = json.loads(Path(layout_path).read_text())
    origin = np.array(doc['world_origin_m'])
    for c in doc['components']:
        site = data.site(c['id']+'_optical_axis')
        c['optical_center_m'] = (site.xpos-origin).tolist()
        c['direction_xyz'] = site.xmat.reshape(3, 3)[:, 2].tolist()
    doc['measurement'] = {'source': 'MuJoCo saved-state FK; not camera metrology',
                          'source_layout_sha256': plan.sha256,
                          'checkpoint_sha256': sha(checkpoint), 'time_s': float(data.time)}
    Path(output).write_text(json.dumps(doc, indent=2)+'\n')
    return doc


def evaluate(document, spec, n):
    import LightPipes as lp
    components = document['components']
    if [c['id'] for c in components] != ['lens_1','pinhole','lens_2']:
        raise ValueError('Unsupported optical inventory')
    centers = [np.asarray(c['optical_center_m'], float) for c in components]
    for c, p, kind in zip(components, centers, ('lens','aperture','lens')):
        if c['type'] != kind:
            raise ValueError('Optical type mismatch')
        normal = np.asarray(c['direction_xyz'], float)
        if (p.shape != (3,) or normal.shape != (3,) or not np.isfinite(p).all() or
                not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1)>1e-6):
            raise ValueError('Invalid optical pose')
        if np.arccos(np.clip(abs(normal[0]),0,1)) > np.deg2rad(1):
            raise ValueError('Outside supported one-degree paraxial orientation range')
    xs = [p[0] for p in centers]
    source_x = spec['source_plane_x_m']
    detector_x = spec['detector_plane_x_m']
    if not source_x <= xs[0] < xs[1] < xs[2] < detector_x:
        raise ValueError('Invalid propagation order')
    wavelength, size = spec['wavelength_m'], spec['window_m']
    f = lp.GaussBeam(lp.Begin(size, wavelength, n), spec['waist_radius_m'])
    f.field *= np.sqrt(spec['input_power_W']/lp.Power(f))
    q = f.mgrid_cartesian
    # LightPipes returns Cartesian Y, X; transverse X maps to world Y here.
    yy, zz = q[1], q[0]
    beam_center = np.asarray(spec['beam_axis_yz_m'])
    records = []
    previous = source_x
    for c, p in zip(components, centers):
        if p[0] > previous:
            f = lp.Forvard(f, float(p[0]-previous))
        dy, dz = p[1:]-beam_center
        normal = np.asarray(c['direction_xyz'])
        # Projection of a circular aperture onto the fixed lab transverse plane.
        # Small tilted-plane longitudinal phase corrections are neglected.
        a, b = yy-dy, zz-dz
        radius = spec['elements'][c['id']]['aperture_radius_m']
        r2 = a*a+b*b+((normal[1]*a+normal[2]*b)/normal[0])**2
        before = float(lp.Power(f))
        f = lp.MultIntensity(f, (r2 <= radius*radius).astype(float))
        if c['type'] == 'lens':
            focal = spec['elements'][c['id']]['focal_length_m']
            f = lp.Lens(f, focal, x_shift=float(dy), y_shift=float(dz))
        records.append({'id': c['id'], 'power_before_W': before,
                        'power_after_W': float(lp.Power(f)),
                        'local_transmission': float(lp.Power(f))/max(before,1e-300)})
        previous = p[0]
    f = lp.Forvard(f, detector_x-previous)
    intensity = lp.Intensity(f)
    total = float(intensity.sum())
    centroid = [float((intensity*v).sum()/max(total,1e-300)) for v in (yy,zz)]
    power = float(lp.Power(f))
    efficiency = power/spec['input_power_W']
    checks = {'transmitted_fraction': efficiency >= spec['acceptance']['min_transmitted_fraction'],
              'output_centroid': np.linalg.norm(centroid) <= spec['acceptance']['max_centroid_offset_m']}
    return {'pass': bool(all(checks.values())), 'checks': {k:bool(v) for k,v in checks.items()},
            'N': n, 'window_m': size, 'power_W': power, 'transmitted_fraction': efficiency,
            'output_centroid_yz_m': centroid, 'stages': records}, intensity


def validate(layout_path, spec_path, output, actual=None):
    import LightPipes as lp
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    nominal = json.loads(Path(layout_path).read_text())
    spec = json.loads(Path(spec_path).read_text())
    if (len(spec['grids'])<2 or any(type(n) is not int or not 256<=n<=4096 for n in spec['grids'])
            or sorted(set(spec['grids'])) != spec['grids']):
        raise ValueError('Need two or more increasing grid sizes in [256,4096]')
    # Bind the optics specification to the original mechanical layout.
    if spec['assembly_layout_sha256'] != sha(layout_path):
        raise ValueError('Optical specification/layout mismatch')
    for name in ['wavelength_m','window_m','waist_radius_m','input_power_W']:
        if not isinstance(spec[name],(int,float)) or not np.isfinite(spec[name]) or spec[name]<=0:
            raise ValueError('Invalid optical parameter '+name)
    for e in spec['elements'].values():
        if any(not np.isfinite(v) or v <= 0 for v in e.values()):
            raise ValueError('Invalid element parameters')
    if actual is not None:
        if actual.get('measurement',{}).get('source_layout_sha256') != sha(layout_path):
            raise ValueError('Actual poses are not bound to this design')
    report = {'engine':'LightPipes', 'version':lp.__version__, 'layout_sha256':sha(layout_path),
              'optical_spec_sha256':sha(spec_path), 'validator_sha256':sha(__file__),
              'scope':'paraxial scalar thin lenses and projected apertures; not full 3D Maxwell or real-hardware validation',
              'parameters_status':'provisional simulation values; not measured instruments', 'cases':{}}
    arrays = {}
    for label, doc in [('design',nominal)]+([] if actual is None else [('actual',actual)]):
        runs=[]
        for n in spec['grids']:
            result, intensity = evaluate(doc,spec,n);runs.append(result)
        relative = abs(runs[-1]['power_W']-runs[-2]['power_W'])/max(runs[-1]['power_W'],runs[-2]['power_W'],1e-30)
        converged = relative <= spec['acceptance']['max_grid_relative_power_difference']
        report['cases'][label]={'pass': all(r['pass'] for r in runs) and converged,
                                'grid_converged': converged, 'grid_power_relative_difference':relative,'runs':runs}
        arrays[label]=intensity
        from .optical_sensor import publish
        measurement_time=doc.get('measurement',{}).get('time_s')
        pinhole=next(s['local_transmission'] for s in runs[-1]['stages'] if s['id']=='pinhole')
        observation=publish(output/(label+'-detector'),intensity,spec,measurement_time,pinhole)
        report['cases'][label]['detector_observation']={
            'file':str(output/(label+'-detector/observation.json')),
            'valid':observation['valid'],'reasons':observation['reasons'],
            'power_W':observation['power_W'],'centroid_yz_m':observation['centroid_yz_m']}
    shifted=deepcopy(nominal)
    shifted['components'][1]['optical_center_m'][1] += .001
    negative,_=evaluate(shifted,spec,spec['grids'][-1])
    report['negative_control']={'pinhole_shift_m':.001,'rejected':not negative['pass'], 'evaluation':negative}
    report['backend_verified']=report['cases']['design']['pass'] and report['negative_control']['rejected']
    report['actual_optical_pass']=None if actual is None else report['cases']['actual']['pass']
    report['actual_observation_valid']=None if actual is None else report['cases']['actual']['detector_observation']['valid']
    if actual is not None:
        (output/'actual-layout.json').write_text(json.dumps(actual,indent=2)+'\n')
    (output/'optical-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    np.savez_compressed(output/'intensity.npz',**arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,len(arrays),figsize=(6*len(arrays),4.5),squeeze=False)
    for ax,(label,a) in zip(axes[0],arrays.items()):
        im=ax.imshow(a,extent=np.array([-1,1,-1,1])*spec['window_m']*500,origin='lower')
        ax.set(title=f'{label}: {report["cases"][label]["runs"][-1]["transmitted_fraction"]:.3g} transmitted',xlabel='world Y offset (mm)',ylabel='world Z offset (mm)')
        fig.colorbar(im,ax=ax,label='Intensity (W/m²)')
    fig.tight_layout();fig.savefig(output/'optical-comparison.png',dpi=150);plt.close(fig)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',type=Path,default=Path('configs/optical_setups/three_mounts.json'))
    p.add_argument('--spec',type=Path,default=Path('configs/optical_setups/three_mounts_optics.json'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    actual=None
    if args.checkpoint:
        # Export in memory; validate() persists it alongside the optical report.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            actual=export_actual(args.layout,args.checkpoint,Path(temp)/'actual.json')
    report=validate(args.layout,args.spec,args.output,actual)
    print(json.dumps({key:report[key] for key in ['backend_verified','actual_optical_pass','actual_observation_valid']}))
    return 0 if report['backend_verified'] and report['actual_optical_pass'] is not False and report['actual_observation_valid'] is not False else 1


if __name__=='__main__':raise SystemExit(main())
