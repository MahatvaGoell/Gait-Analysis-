"""Conservative, auditable gap handling for Subject 06/07 only.

Original finite grid values are immutable. A gap needs two observed endpoints,
a <= 0.50 s endpoint span, and acceptable same-channel blocked holdout error.
No endpoint extrapolation, cross-trial templates, or waveform smoothing.
"""
import numpy as np
from subject08_review import DT, runs
from subject08_segmentation import (load_peaks, movement_onset, cycle_features,
                                    segment_trial, change_window)

# 5% of these floors allows 2 FMG counts, 0.2 CoP units, or 30 vGRF units.
# This engineering tolerance avoids rejecting ordinary quantization noise.
FLOORS = np.r_[np.full(16, 40.), 4., 600., 4., 600.]


def blocked_validation(x, missing_count, floor, target=None):
    """Hide independent observed blocks, predict from their two endpoints.

    Error is RMSE divided by channel p95-p5 (with a digital-unit floor).
    These are empirical interpolation checks, not confidence intervals.
    """
    width = missing_count + 2
    blocks = []
    for a, b in runs(np.isfinite(x)):
        blocks.extend(range(a, b-width+1, width))
    scope='whole_channel'
    if target is not None:
        # Select comparison blocks by endpoints and OUTSIDE context only,
        # never by the hidden values whose prediction errors are being tested.
        scope='matching_endpoint_and_flank_context'
        def descriptor(a,b):
            flank=np.r_[x[max(0,a-10):a+1],x[b:min(len(x),b+11)]]
            if len(flank)<22 or not np.isfinite(flank).all():return None
            return np.array([(x[a]+x[b])/2,x[b]-x[a],np.ptp(flank[:11]),np.ptp(flank[11:])])
        query=descriptor(target[0]-1,target[1])
        if query is None:
            blocks=[]
        else:
            tolerance=max(floor*.05,float(np.nanpercentile(x,95)-np.nanpercentile(x,5))*.05)
            ranked=[]
            for a in blocks:
                desc=descriptor(a,a+width-1)
                if desc is not None and np.all(abs(desc-query)<=2*tolerance):
                    ranked.append((float(np.sum((desc-query)**2)),a))
            blocks=[a for _,a in sorted(ranked)[:40]]
    if len(blocks) > 100:
        blocks = [blocks[i] for i in np.linspace(0, len(blocks)-1, 100).astype(int)]
    if len(blocks) < 20:
        return dict(blocks=len(blocks), p90_normalized_rmse=None, accepted=False,scope=scope)
    scale = max(float(np.nanpercentile(x, 95)-np.nanpercentile(x, 5)), floor)
    errors = []
    for a in blocks:
        truth = x[a+1:a+width-1]
        estimate = np.linspace(x[a], x[a+width-1], width)[1:-1]
        errors.append(float(np.sqrt(np.mean((truth-estimate)**2))/scale))
    p90 = float(np.percentile(errors, 90))
    worst = max(errors)
    return dict(blocks=len(blocks), p90_normalized_rmse=p90,
                max_normalized_rmse=worst, accepted=p90 <= .05 and worst <= .10,scope=scope)


def refine(grid, observed):
    z = observed.copy()
    imputed = np.zeros_like(observed, dtype=bool)
    gaps = []
    for k in range(observed.shape[1]):
        x = observed[:, k]
        cache = {}
        for a, b in runs(~np.isfinite(x)):
            n = b-a
            check = dict(blocks=0, p90_normalized_rmse=None, accepted=False)
            if a == 0 or b == len(x):
                reason = 'entire_channel_absent' if n == len(x) else 'unbounded_edge_gap'
            elif grid[b]-grid[a-1] > .50 + 1e-9:
                reason = 'endpoint_span_exceeds_0.50_seconds'
            else:
                if n not in cache:
                    cache[n] = blocked_validation(x, n, FLOORS[k])
                check = cache[n]
                if not check['accepted']:
                    check=blocked_validation(x,n,FLOORS[k],target=(a,b))
                reason = 'linear_interpolation' if check['accepted'] else 'holdout_check_insufficient_or_error_too_large'
                if check['accepted']:
                    z[a:b, k] = np.interp(grid[a:b], grid[[a-1,b]], x[[a-1,b]])
                    imputed[a:b, k] = True
            gaps.append(dict(channel_index=k, start_seconds=float(grid[a]),
                             end_seconds=float(grid[b-1]), missing_samples=n,
                             endpoint_span_seconds=float(grid[b]-grid[a-1]) if a>0 and b<len(x) else '',
                             method=reason, holdout_blocks=check['blocks'],
                             validation_scope=check.get('scope','not_tested'),
                             holdout_max_normalized_rmse=check.get('max_normalized_rmse'),
                             holdout_p90_normalized_rmse=check['p90_normalized_rmse']))
    assert np.array_equal(z[np.isfinite(observed)], observed[np.isfinite(observed)])
    assert np.array_equal(imputed, ~np.isfinite(observed) & np.isfinite(z))
    return z, imputed, gaps


def gap_aware_change(features, keep=None):
    """Fit observed cycles without filling feature values for missing cycles.

    Requires >=6 usable cycles, >=70% cycle coverage, >=2 observed cycles in
    either regime, and three consecutive usable cycles around the transition.
    """
    if all(f is not None for f in features):
        return change_window(features, keep)
    valid = np.array([f is not None for f in features])
    if valid.sum()<6 or valid.mean()<.70:
        return None
    f = np.array([v for v in features if v is not None])
    center=np.median(f,axis=0)
    scale=np.maximum(np.percentile(f,90,axis=0)-np.percentile(f,10,axis=0),
                     np.r_[np.full(32,2.), [.2,30,.2,30,.2,30,.2,30], .05])
    x=np.full((len(features),41),np.nan)
    x[valid]=(f-center)/scale
    weights=np.r_[np.full(32,1/96),np.full(8,1/24),1/3]
    if keep is not None:
        weights*=keep; weights/=weights.sum()
    def cost(v):
        return float(np.sum((v-v.mean(axis=0))**2*weights))
    total=cost(x[valid]); trials=[]
    for k in range(2,len(x)-2):
        if not valid[k-1:k+2].all() or valid[:k].sum()<2 or valid[k+1:].sum()<2:
            continue
        a=x[:k][valid[:k]]; b=x[k+1:][valid[k+1:]]
        mid=(a.mean(axis=0)+b.mean(axis=0))/2
        trials.append((k,cost(a)+cost(b)+.5*float(np.sum((x[k]-mid)**2*weights))))
    if not trials:return None
    k,best=min(trials,key=lambda q:q[1])
    return dict(index=k,improvement=1-best/max(total,1e-9),
                alternatives=[j for j,c in trials if c<=best+max(.10*total,.02)])


def segment_refined(grid, z, imputed):
    # Keep the established estimator when its complete-cycle conditions hold.
    phases,info,landmarks=segment_trial(grid,z)
    info['excluded_incomplete_cycles']=0
    if phases:
        # Do not accept a cycle whose evidence is predominantly interpolated.
        pk,_=load_peaks(z[:,17 if info['reference_limb']=='L' else 19])
        pk=[p for p in pk if info['gi_end_seconds']<=p*DT<=info['gt_start_seconds']]
        if not any(np.any(imputed[a:b].mean(axis=0)>.10) for a,b in zip(pk[:-1],pk[1:])):
            return phases,info,landmarks
        phases=[]
        _,info,_=segment_trial(grid,np.ones_like(z))
        info['excluded_incomplete_cycles']=0
        info['flags']='insufficient complete strides or missing data; interpolated-cycle evidence excluded'
    if 'insufficient complete strides or missing data' not in info['flags']:
        return phases,info,landmarks
    left,_=load_peaks(z[:,17]); right,_=load_peaks(z[:,19])
    info['load_peaks_left']=len(left);info['load_peaks_right']=len(right)
    ref='L' if len(left)>=len(right) else 'R'; peaks=left if ref=='L' else right
    first=min(left[0],right[0]); gt=min(left[-1],right[-1])
    onset=movement_onset(z,first,float(np.median(np.diff(peaks))))
    usable=[p for p in peaks if first<=p<=gt]
    features=cycle_features(z,usable)
    # A cycle dominated by interpolated data is not independent observed evidence.
    for j,(a,b) in enumerate(zip(usable[:-1],usable[1:])):
        if np.any(imputed[a:b].mean(axis=0)>.10):features[j]=None
    result=gap_aware_change(features)
    info['excluded_incomplete_cycles']=sum(v is None for v in features)
    if result is None:return phases,info,landmarks
    k=result['index']; start,end=usable[k:k+2]
    changes=[]
    for a,b in ((0,32),(32,40),(40,41)):
        keep=np.ones(41);keep[a:b]=0
        changes.append(gap_aware_change(features,keep)['index'])
    boundaries=[0,onset,first,start,end,gt,len(grid)]
    assert all(a<b for a,b in zip(boundaries[:-1],boundaries[1:]))
    phases=[dict(phase=label,start_index=int(a),end_index=int(b),start_seconds=round(a*DT,2),
                 end_seconds=round(float(grid[-1]) if b==len(grid) else b*DT,2),status='provisional')
            for label,a,b in zip(('QS','GI','SSSW','SLT','SSLW','GT'),boundaries[:-1],boundaries[1:])]
    alternatives=result['alternatives']+changes
    # Missing-cycle fits remain weak even when the numerical contrast is large.
    info.update(status='provisional six-phase fit',onset_seconds=round(onset*DT,2),
                gi_end_seconds=round(first*DT,2),slt_start_seconds=round(start*DT,2),slt_end_seconds=round(end*DT,2),
                gt_start_seconds=round(gt*DT,2),transition_support='weak/ambiguous pattern evidence',
                transition_improvement=round(result['improvement'],4),reference_limb=ref,
                alternative_transition_start_seconds=round(usable[min(alternatives)]*DT,2),
                alternative_transition_end_seconds=round(usable[max(alternatives)+1]*DT,2),
                flags='Provisional gap-aware fit; incomplete cycles excluded, not reconstructed; short-to-long order assumed; SLT timing ambiguous; pressure extrema are not heel strikes')
    return phases,info,landmarks
