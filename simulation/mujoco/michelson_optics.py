"""Balanced two-arm Michelson scalar model driven by an actual mirror normal.

The adjustable end mirror is illuminated along +Y. Its return is transmitted
through the splitter to a fixed detector; the reference return is reflected into
that same port. Ideal 50/50 splitter and flat mirrors; no polarization.
"""
import json
from pathlib import Path
import numpy as np
import LightPipes as lp
from scipy.ndimage import shift

SPEC=json.loads((Path(__file__).resolve().parents[2]/'configs/optical_setups/michelson_bench.json').read_text())
WAVELENGTH=SPEC['wavelength_m']
SIZE=SPEC['detector_window_m']
POWER=SPEC['input_power_W']
BS=np.array(SPEC['splitter_center_m']);M1=np.array(SPEC['reference_mirror_center_m'])
M2=np.array(SPEC['adjustable_mirror_center_m']);DETECTOR=np.array(SPEC['detector_center_m'])
ARM1=float(np.linalg.norm(M1-BS));ARM2=float(np.linalg.norm(M2-BS));OUTPUT=float(np.linalg.norm(DETECTOR-BS))
if not (np.allclose(M1-BS,[ARM1,0,0]) and np.allclose(M2-BS,[0,ARM2,0]) and np.allclose(DETECTOR-BS,[0,-OUTPUT,0])):
    raise ValueError('Michelson bench requires +X reference, +Y adjustable, -Y detector ports')


def evaluate(normal,n=256,capture_folder=None):
    normal=np.asarray(normal,dtype=float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1)>1e-6:
        raise ValueError('Invalid mirror normal')
    incoming=np.array([0.,1.,0.]);returned=incoming-2*(incoming@normal)*normal
    if returned[1]>=-.9:raise ValueError('Mirror return outside paraxial detector port')
    length=ARM2+(ARM2+OUTPUT)/(-returned[1])
    spot=returned[[0,2]]*(ARM2+OUTPUT)/(-returned[1])
    base=lp.GaussBeam(lp.Begin(SIZE,WAVELENGTH,n),SPEC['gaussian_waist_m'])
    base.field*=np.sqrt(POWER/lp.Power(base))
    reference=lp.Forvard(base,2*ARM1+OUTPUT)
    measured=lp.Forvard(base,float(length))
    offset=spot/(SIZE/n)
    measured.field=(shift(measured.field.real,[offset[1],offset[0]],order=1,mode='constant')+
                    1j*shift(measured.field.imag,[offset[1],offset[0]],order=1,mode='constant'))
    measured=lp.Tilt(measured,-float(returned[0]),-float(returned[2]))
    # Two splitter encounters: output field amplitude is 1/2 for each arm.
    a,b=reference.field*.5,measured.field*.5
    phases=np.arange(SPEC['reference_phase_samples'])*2*np.pi/SPEC['reference_phase_samples']
    # Ideal detector pixels are the only inputs to fringe-fit feedback.
    images=[abs(a+np.exp(1j*p)*b)**2 for p in phases]
    full_scale=SPEC['detector_full_scale_W_m2']
    counts=[np.rint(np.clip(i/full_scale,0,1)*65535).astype(np.uint16) for i in images]
    if capture_folder is not None:
        import cv2
        from pathlib import Path
        folder=Path(capture_folder);folder.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(counts):
            if not cv2.imwrite(str(folder/f'phase-{i:02d}-detector16.png'),c):raise RuntimeError('Phase detector write failed')
    powers=np.array([float(c.astype(float).sum())*full_scale/65535*(SIZE/n)**2 for c in counts])
    design=np.column_stack([np.ones(len(phases)),np.cos(phases),np.sin(phases)])
    fit=np.linalg.lstsq(design,powers,rcond=None)[0]
    visibility=float(np.hypot(fit[1],fit[2])/max(fit[0],1e-30))
    valid=bool(fit[0]>1e-9 and not any(np.any(i>=full_scale) for i in images))
    field=lp.Field.copy(reference);field.field=a+b
    return {'valid':valid,'visibility':float(np.clip(visibility,0,1)),'raw_visibility_fit':visibility,'mean_port_power_W':float(fit[0]),'full_scale_W_m2':full_scale,'window_m':SIZE,'grid_size':n,
            'phase_scan_power_W':powers.tolist(),
            'objective_source':'8 reference-phase samples of ideal 16-bit detector images; integrated-port modulation visibility; no normal/angle error supplied to policy',
            'model':'Michelson ideal 50/50 splitter, reflected chief ray and paraxial coherent propagation',
            'assumptions':'scalar coherent 632.8nm, 1mW source, 0.5mm waist, balanced 204mm arms; provisional'},lp.Intensity(field)


def from_state(model,data,n=256,capture_folder=None):
    site=data.site('mirror_optical_axis')
    return evaluate(site.xmat.reshape(3,3)[:,2],n,capture_folder)


class FringeSearch:
    """Coordinate hill-climb; sees camera visibility, never mirror state."""
    def __init__(self):
        self.commands=np.zeros(2);self.sign=np.ones(2);self.step=np.full(2,.1)
        self.best_score=None;self.best_commands=np.zeros(2)
        self.axis=0;self.actions=0

    def decide(self,observation):
        if not observation['valid']:return 'invalid',self.commands.copy()
        score=observation['visibility']
        if score>=SPEC['accept_visibility']:return 'accept',self.commands.copy()
        if self.actions>=30:return 'limit',self.commands.copy()
        if self.best_score is None or score>self.best_score+1e-5:
            self.best_score=score;self.best_commands=self.commands.copy()
        elif self.actions:
            self.sign[self.axis]*=-1
            if score<self.best_score-1e-4:self.step[self.axis]=max(.0125,self.step[self.axis]*.5)
        self.axis=self.actions%2
        self.commands=self.best_commands.copy()
        self.commands[self.axis]+=self.sign[self.axis]*self.step[self.axis]
        self.commands=np.clip(self.commands,-1.,1.)
        self.actions+=1
        return 'move',self.commands.copy()


def report_plot(folder,result,config=None):
    """Export auditable phase curves and the measured alignment history."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    folder=Path(folder);history=result['history'];config=SPEC if config is None else config
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    axes[0].plot([x['time_s'] for x in history],[100*x['visibility'] for x in history],'o-')
    axes[0].axhline(95,color='gray',linestyle='--',linewidth=1)
    axes[0].set(xlabel='Simulation time (s)',ylabel='Integrated modulation visibility (%)',ylim=(0,101))
    phase=np.arange(config['reference_phase_samples'])*2*np.pi/config['reference_phase_samples']
    for name,obs in [('Initial',history[0]),('After tool release',result['after_release'])]:
        axes[1].plot(phase,np.array(obs['phase_scan_power_W'])*1e6,'o-',label=name)
    axes[1].set(xlabel='Ideal reference phase scan (rad)',ylabel='Detector port power (uW)');axes[1].legend()
    image=cv2.imread(str(folder/'optical-initial/phase-00-detector16.png'),cv2.IMREAD_UNCHANGED)
    axes[2].imshow(image,origin='lower',extent=np.array([-1,1,-1,1])*config['detector_window_m']*500,cmap='inferno',vmin=0,vmax=65535)
    axes[2].set(xlabel='Detector X (mm)',ylabel='Detector Z (mm)',title='Initial phase-0 detector (fixed gain)')
    fig.suptitle('Michelson contact adjustment | ideal phase-scanned detector | physical contact adjustment')
    fig.savefig(folder/'michelson-verification.png',dpi=150);plt.close(fig)
