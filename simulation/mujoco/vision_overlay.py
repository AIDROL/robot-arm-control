"""Image-only diagnostic annotations. These detections never command the robot."""
import cv2
import numpy as np
from .global_camera import DICTIONARY, TOP_EDGE, SIDE_EDGE


def annotate(rgb, K, camera, size=(640,360), detections=None, background=None):
    """Detect at sensor resolution, draw in preview pixels; RGB/depth stay aligned."""
    height,width=rgb.shape[:2]
    scale=np.array(size)/[width,height]
    canvas=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR) if background is None else background,size)
    K=np.asarray(K,dtype=float)
    Ks=K.copy();Ks[0]*=scale[0];Ks[1]*=scale[1]
    if detections is None:
        parameters=cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
        corners,ids,rejected=cv2.aruco.ArucoDetector(DICTIONARY,parameters).detectMarkers(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
        detections=[]
        for quad,marker in zip(corners,[] if ids is None else ids.flatten()):
            points=quad.reshape(4,2).astype(float);marker=int(marker)
            edge=TOP_EDGE if 0<=marker<5 else SIDE_EDGE if 10<=marker<15 else None
            record={'id':marker,'corners_px':points.tolist(),'pose_valid':False}
            if edge is not None:
                h=edge/2;square=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],float)
                _,rvecs,tvecs,_=cv2.solvePnPGeneric(square,points,K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE_SQUARE)
                choices=[]
                for rv,tv in zip(rvecs,tvecs):
                    if not np.isfinite(rv).all() or not np.isfinite(tv).all() or tv[2,0]<=0:continue
                    projection=cv2.projectPoints(square,rv,tv,K,np.zeros(5))[0].reshape(4,2)
                    error=float(np.sqrt(np.mean(np.sum((projection-points)**2,axis=1))))
                    choices.append((error,rv,tv))
                if choices:
                    error,rv,tv=min(choices,key=lambda x:x[0])
                    record.update(pose_valid=True,rvec=rv.ravel().tolist(),tvec_m=tv.ravel().tolist(),reprojection_error_px=error,edge_m=edge)
            detections.append(record)
    for record in detections:
        pts=np.asarray(record['corners_px'])*scale
        small = np.max(np.ptp(pts,axis=0)) < 40
        cv2.polylines(canvas,[np.rint(pts).astype(np.int32)],True,(80,230,80),1,cv2.LINE_AA)
        for i,p in enumerate(pts):
            xy=tuple(np.rint(p).astype(int))
            cv2.circle(canvas,xy,3,(0,190,255),-1,cv2.LINE_AA)
            dx,dy = [(-16,-12),(6,-12),(6,18),(-16,18)][i] if small else (4,-4)
            cv2.putText(canvas,str(i),(xy[0]+dx,xy[1]+dy),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,230,255),1,cv2.LINE_AA)
        title=f'ID {record["id"]}'
        if record['pose_valid']:
            rv=np.array(record['rvec']);tv=np.array(record['tvec_m'])
            cv2.drawFrameAxes(canvas,Ks,np.zeros(5),rv,tv,record['edge_m']*.6,1)
            title+=f' err={record["reprojection_error_px"]:.2f}px'
        xy=tuple(np.rint(pts.mean(axis=0)).astype(int))
        if small:
            cv2.putText(canvas,f'ID {record["id"]}',(xy[0]-16,xy[1]+32),cv2.FONT_HERSHEY_SIMPLEX,.38,(80,255,150),1,cv2.LINE_AA)
        else:
            cv2.putText(canvas,title,(xy[0]+8,xy[1]+15),cv2.FONT_HERSHEY_SIMPLEX,.36,(80,255,150),1,cv2.LINE_AA)
    errors=', '.join(f'{r["id"]}:{r["reprojection_error_px"]:.2f}' if r['pose_valid'] else f'{r["id"]}:no pose' for r in detections)
    lines=[f'{camera} {width}x{height} | corner 0-3 | ID/err(px): {errors or "none"}',
           f'SIM calibration | fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} D=0',
           'Diagnostic PnP axes: X red / Y green / Z blue; not control acceptance']
    for i,line in enumerate(lines):
        y=size[1]-38+i*16
        cv2.rectangle(canvas,(0,y-12),(size[0],y+3),(20,20,20),-1)
        cv2.putText(canvas,line,(5,y),cv2.FONT_HERSHEY_SIMPLEX,.34,(230,230,230),1,cv2.LINE_AA)
    return canvas,detections
