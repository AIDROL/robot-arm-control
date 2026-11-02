"""Michelson chief rays and coherent fields from all five achieved module poses."""
from pathlib import Path
import cv2
import numpy as np
import LightPipes as lp
from scipy.ndimage import shift

ROLES={'laser':'lens_1','splitter':'pinhole','reference_mirror':'lens_2','detector':'beam_splitter','adjustable_mirror':'mirror'}


def reflect(direction,normal):
    result=direction-2*np.dot(direction,normal)*normal
    return result/np.linalg.norm(result)


def intersect(origin,direction,center,normal,radius=None):
    cosine=float(direction@normal)
    if abs(cosine)<1e-8:raise ValueError('RAY_PARALLEL_TO_OPTIC')
    distance=float((center-origin)@normal/cosine)
    if distance<=1e-7:raise ValueError('OPTIC_BEHIND_RAY')
    point=origin+distance*direction
    if radius is not None and np.linalg.norm(point-center)>radius:raise ValueError('CHIEF_RAY_MISSED_CLEAR_APERTURE')
    return point,distance


def trace(poses):
    source,splitter,reference,adjustable,detector=(poses[r] for r in ('laser','splitter','reference_mirror','adjustable_mirror','detector'))
    direction=source[:3,2]
    first,l0=intersect(source[:3,3],direction,splitter[:3,3],splitter[:3,2],.011)
    paths=[]
    for mirror,incoming,second_reflection in [(reference,direction,True),(adjustable,reflect(direction,splitter[:3,2]),False)]:
        point,l1=intersect(first,incoming,mirror[:3,3],mirror[:3,2],.011)
        returned=reflect(incoming,mirror[:3,2])
        second,l2=intersect(point,returned,splitter[:3,3],splitter[:3,2],.011)
        outgoing=reflect(returned,splitter[:3,2]) if second_reflection else returned
        hit,l3=intersect(second,outgoing,detector[:3,3],detector[:3,2])
        offset=hit-detector[:3,3];u=detector[:3,0];v=-detector[:3,1]
        cosine=abs(float(outgoing@detector[:3,2]))
        if cosine<.99:raise ValueError('NONPARAXIAL_DETECTOR_RAY')
        paths.append({'length_m':l0+l1+l2+l3,'offset_m':[float(offset@u),float(offset@v)],
            'slope':[float(outgoing@u/cosine),float(outgoing@v/cosine)],
            'points_m':[source[:3,3].tolist(),first.tolist(),point.tolist(),second.tolist(),hit.tolist()]})
    return paths


def from_state(model,data,n=256,capture_folder=None,config=None):
    poses={}
    for role,name in ROLES.items():
        site=data.site(name+'_optical_axis');T=np.eye(4)
        T[:3,:3]=site.xmat.reshape(3,3);T[:3,3]=site.xpos;poses[role]=T
    try:paths=trace(poses)
    except ValueError as error:
        return {'valid':False,'visibility':0.,'reason':str(error),'phase_scan_power_W':[],
            'objective_source':'actual module planes and detector pixels'},np.zeros((n,n))
    size=config['detector_window_m'];power=config['input_power_W'];full=config['detector_full_scale_W_m2'];dx=size/n
    base=lp.GaussBeam(lp.Begin(size,config['wavelength_m'],n),config['gaussian_waist_m'])
    base.field*=np.sqrt(power/lp.Power(base))
    fields=[]
    for path in paths:
        field=lp.Forvard(base,path['length_m']);offset=np.array(path['offset_m'])/dx
        field.field=(shift(field.field.real,[offset[1],offset[0]],order=1,mode='constant')+
            1j*shift(field.field.imag,[offset[1],offset[0]],order=1,mode='constant'))
        field=lp.Tilt(field,-path['slope'][0],-path['slope'][1]);fields.append(field.field*.5)
    phases=np.arange(config['reference_phase_samples'])*2*np.pi/config['reference_phase_samples']
    images=[abs(fields[0]+np.exp(1j*p)*fields[1])**2 for p in phases]
    counts=[np.rint(np.clip(i/full,0,1)*65535).astype(np.uint16) for i in images]
    if capture_folder is not None:
        folder=Path(capture_folder);folder.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(counts):
            if not cv2.imwrite(str(folder/f'phase-{i:02d}-detector16.png'),c):raise RuntimeError('Phase image write failed')
    powers=np.array([c.astype(float).sum()*full/65535*dx**2 for c in counts])
    fit=np.linalg.lstsq(np.column_stack([np.ones(len(phases)),np.cos(phases),np.sin(phases)]),powers,rcond=None)[0]
    cross=sum(c.astype(float)*np.exp(-1j*p) for c,p in zip(counts,phases))/len(phases)
    # Phase differences are obtained from the detector phase scan, not ray truth.
    k=2*np.pi/config['wavelength_m']
    gradient=[float(np.angle(np.sum(np.conj(cross[:,:-1])*cross[:,1:]))/dx/k),
              float(np.angle(np.sum(np.conj(cross[:-1,:])*cross[1:,:]))/dx/k)]
    visibility=float(np.hypot(fit[1],fit[2])/max(fit[0],1e-30))
    valid=bool(fit[0]>.2*power and not any(np.any(i>=full) for i in images))
    return {'valid':valid,'visibility':float(np.clip(visibility,0,1)),'raw_visibility_fit':visibility,
        'mean_port_power_W':float(fit[0]),'phase_scan_power_W':powers.tolist(),'grid_size':n,'paths':paths,
        'window_m':size,'full_scale_W_m2':full,'relative_tilt_from_pixels_rad':gradient,
        'objective_source':'phase-scanned quantized detector images; all achieved module planes drive the optical plant',
        'model':'actual-plane chief rays + scalar paraxial coherent fields; ideal reference-phase scan; chief-ray clear-aperture gates only'},images[0]


class PhaseGradientSearch:
    """Pixel-demodulated relative wavefront tilt drives the two calibrated knobs."""
    def __init__(self,ratio=.0016):
        self.commands=np.zeros(2);self.ratio=ratio;self.actions=0

    def decide(self,observation):
        if not observation['valid']:return 'invalid',self.commands.copy()
        error=np.array(observation['relative_tilt_from_pixels_rad'])
        # Leave optical margin for the small measured extraction disturbance;
        # the final task still requires a fresh >=95% result after tool return.
        if observation['visibility']>=.99 and np.linalg.norm(error)<5e-6:return 'accept',self.commands.copy()
        if self.actions>=60:return 'limit',self.commands.copy()
        if not np.isfinite(error).all():return 'invalid',self.commands.copy()
        delta=np.array([error[1],-error[0]])/(2*self.ratio)
        self.commands+=np.clip(.6*delta,-.5,.5)
        self.commands=np.clip(self.commands,-10.,10.);self.actions+=1
        return 'move',self.commands.copy()
