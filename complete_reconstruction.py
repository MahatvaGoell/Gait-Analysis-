"""Explicit model-completed views; never recovered measurements or ground truth.

Preserve every finite source-grid value. Predict missing groups from other
concurrent sensors with ridge regression; select against temporal interpolation
on blocked holdouts. Entirely absent groups use same-subject donor recordings.
"""
import numpy as np
from subject08_review import DT, runs
from subject08_segmentation import segment_trial, load_peaks, movement_onset, cycle_features, change_window

GROUPS={'FMG':list(range(16)), 'L_insole':[16,17], 'R_insole':[18,19]}


def temporal(x):
    result=x.copy();t=np.arange(len(x))
    for k in range(x.shape[1]):
        valid=np.isfinite(x[:,k])
        if valid.any():result[:,k]=np.interp(t,t[valid],x[valid,k])
    return result


def features(z, columns):
    x=temporal(z[:,columns])
    if not np.isfinite(x).all():return None
    # These predictors never include any channels from the target group.
    t=np.arange(len(x))
    return np.column_stack([x,x[np.clip(t-10,0,len(x)-1)],x[np.clip(t+10,0,len(x)-1)]])


def design(x,center,scale):
    q=np.clip((x-center)/scale,-8,8)
    return np.column_stack([np.ones(len(x)),q,q*q,np.tanh(q)])


def train_predict(x,y,test):
    idx=np.linspace(0,len(x)-1,min(len(x),5000)).astype(int)
    x=x[idx];y=y[idx]
    center=np.median(x,axis=0);scale=np.maximum(np.std(x,axis=0),1.)
    a=design(x,center,scale);b=design(test,center,scale)
    penalty=np.eye(a.shape[1])*10;penalty[0,0]=1e-6
    weights=np.linalg.solve(a.T@a+penalty,a.T@y)
    pred=b@weights
    # Bounds constrain estimates only; recorded values are never clipped.
    return np.clip(pred,np.min(y,axis=0),np.max(y,axis=0))


def complete_trial(trial, trials):
    record,number,grid,observed,*_=trial
    completed=observed.copy();reports=[]
    for group,target in GROUPS.items():
        missing=~np.isfinite(observed[:,target])
        if not missing.any():continue
        predictors=[k for k in range(20) if k not in target and np.isfinite(observed[:,k]).any()]
        x=features(observed,predictors)
        known=np.all(np.isfinite(observed[:,target]),axis=1)
        y=observed[:,target]
        donor_names=[];model_score=None;linear_score=None;validation_points=0
        source='within_recording';method='temporal_interpolation_with_endpoint_hold'
        estimate=temporal(y)
        if known.sum()>=200 and x is not None:
            # Four observed one-second blocks, including early/late coverage.
            eligible=[]
            for a,b in runs(known):eligible.extend(range(a,b-99,100))
            held=np.zeros(len(grid),bool)
            if len(eligible)>=8:
                for j in np.linspace(0,len(eligible)-1,4).astype(int):
                    a=eligible[j];held[a:a+100]=True
                training=known & ~held
                hidden=y.copy();hidden[held]=np.nan
                temporal_prediction=temporal(hidden)[held]
                prediction=train_predict(x[training],y[training],x[held])
                scale=np.maximum(np.percentile(y[known],95,axis=0)-np.percentile(y[known],5,axis=0),1.)
                model_score=float(np.sqrt(np.mean(((prediction-y[held])/scale)**2)))
                linear_score=float(np.sqrt(np.mean(((temporal_prediction-y[held])/scale)**2)))
                validation_points=int(held.sum())
                if model_score<linear_score:
                    estimate=train_predict(x[known],y[known],x)
                    method='cross_sensor_ridge_within_recording'
        elif x is not None:
            # A missing whole stream cannot be learned from its own trial.
            # Prefer sibling intervals of the same recording, else this subject.
            donors=[t for t in trials if (t[0],t[1])!=(record,number)]
            siblings=[t for t in donors if t[0]==record]
            def samples(candidates):
                xx=[];yy=[];names=[]
                for d in candidates:
                    dz=d[3];dx=features(dz,predictors)
                    if dx is None:continue
                    ok=np.all(np.isfinite(dz[:,target]),axis=1) & np.all(np.isfinite(dz[:,predictors]),axis=1)
                    ids=np.flatnonzero(ok)[::5]
                    if len(ids)<40:continue
                    xx.append(dx[ids]);yy.append(dz[np.ix_(ids,target)]);names.append(f'{d[0]}/{d[1]:02d}')
                return xx,yy,names
            xx,yy,donor_names=samples(siblings)
            if sum(len(a) for a in xx)<400:xx,yy,donor_names=samples(donors)
            if xx:
                tx=np.concatenate(xx);ty=np.concatenate(yy)
                # Entire donor intervals, not random rows, form the test set.
                if len(xx)>=3:
                    test_ids=np.arange(len(xx))[::3];train_ids=[j for j in range(len(xx)) if j not in test_ids]
                    vx=np.concatenate([xx[j] for j in test_ids]);vy=np.concatenate([yy[j] for j in test_ids])
                    prediction=train_predict(np.concatenate([xx[j] for j in train_ids]),np.concatenate([yy[j] for j in train_ids]),vx)
                    scale=np.maximum(np.percentile(ty,95,axis=0)-np.percentile(ty,5,axis=0),1.)
                    model_score=float(np.sqrt(np.mean(((prediction-vy)/scale)**2)))
                    validation_points=len(vy)
                estimate=train_predict(tx,ty,x)
                method='cross_sensor_ridge_same_subject_donors'
                source='same_subject_other_intervals_NOT_measured_in_target'
        # A completely unavailable predictor combination would use a plainly
        # labelled subject-level prior, never disguised as an observed waveform.
        for j,k in enumerate(target):
            if not np.isfinite(estimate[:,j]).all():
                values=np.concatenate([d[3][:,k][np.isfinite(d[3][:,k])] for d in trials])
                if not len(values):raise ValueError(f'No subject data at all for channel {k}')
                estimate[:,j]=np.nanmedian(values)
                method='constant_same_subject_prior_no_waveform_information'
            for a,b in runs(missing[:,j]):
                pred=estimate[a:b,j].copy()
                # Boundary matching avoids a discontinuity at the splice;
                # this correction is applied solely inside estimated regions.
                if method.startswith('cross_sensor'):
                    if a>0 and b<len(grid):
                        correction=np.linspace(y[a-1,j]-estimate[a-1,j],y[b,j]-estimate[b,j],b-a+2)[1:-1]
                        pred+=correction
                    elif a>0:pred+=(y[a-1,j]-estimate[a-1,j])*np.exp(-np.arange(1,b-a+1)/50)
                    elif b<len(grid):pred+=(y[b,j]-estimate[b,j])*np.exp(-np.arange(b-a,0,-1)/50)
                    pred=np.maximum(pred,0) if k<16 or k in (17,19) else pred
                completed[a:b,k]=pred
                reports.append(dict(channel_index=k,start_seconds=float(grid[a]),end_seconds=float(grid[b-1]),
                    missing_samples=int(b-a),method=method,training_source=source,
                    donors=';'.join(donor_names),validation_points=validation_points,
                    model_holdout_normalized_rmse=model_score,temporal_holdout_normalized_rmse=linear_score,
                    warning='ESTIMATE, not recovered measurement; long gaps/absent streams unverified'))
    mask=~np.isfinite(observed)
    assert np.isfinite(completed).all()
    assert np.array_equal(completed[~mask],observed[~mask])
    return completed,mask,reports


def complete_segments(grid,z):
    phases,info,events=segment_trial(grid,z)
    if not phases:
        left,_=load_peaks(z[:,17]);right,_=load_peaks(z[:,19])
        peaks=left if len(left)>=len(right) else right
        if len(peaks)>=2:
            first,last=peaks[0],peaks[-1]
            stride=float(np.median(np.diff(peaks)))
            onset=movement_onset(z,first,stride)
            if onset is None:onset=max(0,int(first-.5*stride))
            # No six-stage claim if too few cycles/insufficient ending evidence.
            points=[(0,onset,'QS'),(onset,first,'GI'),(first,len(grid),'WALK')]
            result=change_window(cycle_features(z,peaks))
            if result is not None and grid[-1]-last*DT>=.8:
                k=result['index'];a,b=peaks[k:k+2]
                if onset<first<a<b<last:
                    points=[(0,onset,'QS'),(onset,first,'GI'),(first,a,'SSSW'),(a,b,'SLT'),(b,last,'SSLW'),(last,len(grid),'GT')]
                    info.update(slt_start_seconds=round(a*DT,2),slt_end_seconds=round(b*DT,2),gt_start_seconds=round(last*DT,2),
                                alternative_transition_start_seconds=round(peaks[min(result['alternatives'])]*DT,2),
                                alternative_transition_end_seconds=round(peaks[max(result['alternatives'])+1]*DT,2))
            phases=[dict(phase=label,start_index=int(a),end_index=int(b),start_seconds=round(a*DT,2),
                end_seconds=round(float(grid[-1]) if b==len(grid) else b*DT,2),status='model_assisted_provisional')
                for a,b,label in points if b>a]
            info.update(onset_seconds=round(onset*DT,2),gi_end_seconds=round(first*DT,2),
                        transition_support='model-assisted; unverified')
        else:
            phases=[dict(phase='UNKNOWN',start_index=0,end_index=len(grid),start_seconds=0.,end_seconds=float(grid[-1]),status='unknown')]
    six=[p['phase'] for p in phases]==['QS','GI','SSSW','SLT','SSLW','GT']
    info.update(status='model-assisted provisional six-phase fit' if six else 'partial phase annotation',
                view_mode='model_completed',phase_count=len(phases),complete_six_phase=six,
                flags=info['flags']+'; reconstructed values can affect boundaries; absent phases are not invented',
                excluded_incomplete_cycles=0)
    return phases,info,events
