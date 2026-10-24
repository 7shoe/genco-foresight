import numpy as np
import torch

from scoring import UNITS,ensemble_energy,evaluate,fair_energy_score,marginal_scores,merge_origin_scores,variogram_pairs,variogram_score
from training.forecaster import design,fit_ridge,predict_ridge,prefix_fit_from_moments


def test_fair_crps_and_energy_hand_calculation_and_point_mass():
    samples=np.array([[[0.]],[[2.]]]);target=np.array([[3.]])
    scores=marginal_scores(samples,target)
    assert scores['crps'].item()==1.
    assert ensemble_energy(samples,target,np.ones_like(target),np.ones_like(target,dtype=bool))==1.
    point=marginal_scores(samples[:1],target)
    assert point['crps'].item()==3. and point['width_90'].item()==0.
    assert point['coverage_90'].item()==0.


def test_joint_energy_matches_training_and_masks_unknowns():
    rng=np.random.default_rng(7);x=rng.normal(size=(8,3,4));y=rng.normal(size=(3,4))
    y[0,0]=np.nan;scale=np.arange(1,13).reshape(3,4)
    mask=np.isfinite(y)
    got=ensemble_energy(x,y,scale,mask)
    expected=fair_energy_score(torch.tensor(x.reshape(8,-1)),torch.tensor(y.ravel()),torch.tensor(scale.ravel()))
    np.testing.assert_allclose(got,float(expected),rtol=1e-12)
    x[:,0,0]=np.nan
    assert ensemble_energy(x,y,scale,mask)==got


def test_variogram_distinguishes_dependence_with_equal_marginals():
    correlated=np.array([[0.,0.],[2.,2.]])
    anticorrelated=np.array([[0.,2.],[2.,0.]])
    truth=np.array([1.,1.]);pairs=np.array([[0,1]])
    assert variogram_score(correlated,truth,pairs,np.ones(2))['score']==0.
    assert variogram_score(anticorrelated,truth,pairs,np.ones(2))['score']>1.


def test_ridge_design_does_not_read_future_operating_states():
    rng=np.random.default_rng(8)
    data={'covariates':rng.normal(size=(120,30,6)), 'calendar':rng.normal(size=(120,5))}
    before=design(data,[72],np.array([1,3,7]))
    data['covariates'][72:]=999.
    np.testing.assert_array_equal(before,design(data,[72],np.array([1,3,7])))


def test_incremental_prefix_ridge_matches_direct_fit_and_excludes_future_targets():
    rng=np.random.default_rng(19);x=rng.normal(size=(40,2,3,5));y=rng.normal(size=(40,2,3))
    cutoff=25;z=x[:cutoff];t=y[:cutoff]
    fit=prefix_fit_from_moments(cutoff,z.sum(0),np.einsum('nbhf,nbhg->bhfg',z,z),
        t.sum(0),np.einsum('nbhf,nbh->bhf',z,t))
    expected=fit_ridge(z,t)
    np.testing.assert_allclose(predict_ridge(fit,x),predict_ridge(expected,x),atol=1e-12)
    y[cutoff:]=10000.
    np.testing.assert_allclose(predict_ridge(fit,x),predict_ridge(fit_ridge(x[:cutoff],y[:cutoff]),x),atol=1e-12)


def test_full_score_report_counts_and_json_round_trip():
    import json
    rng=np.random.default_rng(33)
    targets={k:np.array([[0.,np.nan],[2.,3.]]) for k in UNITS}
    samples={k:rng.normal(size=(4,2,2)) for k in UNITS}
    masks={k:np.isfinite(v) for k,v in targets.items()}
    scales={'features':{k:{'joint_score_mask':[True,True]} for k in UNITS if k!='loading'},
            'scale_vector_one_hour':[1.]*20}
    pairs=variogram_pairs(scales,{'branch':np.array([[0,1]])},horizon=2)
    report,detail=evaluate(samples,targets,masks,{'scales':scales,'variogram_pairs':pairs,'mode':'S','origin_id':'test','seed':42})
    assert report['joint_observed_coordinates']==30
    combined=merge_origin_scores([report,report])
    assert combined['variables']['vm']['count']==6
    assert sum(combined['variables']['vm']['pit_histogram'])==6
    assert json.loads(json.dumps(combined,allow_nan=False))['origins']==2
