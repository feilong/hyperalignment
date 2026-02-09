#!/usr/bin/env python
# coding: utf-8
'''
Script to call for finding the transformation to template using either ridge regression (warp) or generalized procrustes.
To-do: 
 - add back the searchlight hyperalignment for cortical
 - need to add compatibility/switch to new template naming
 - need to add support when the template is made from multiple targets. Is it the same as concatenating the different templates if the data matrices were zscored separately?
 - make a symlink to the template directory such that it follows /templates/targets-{target}_{roi} with targets specifying the onavg-ico8, aseg, TianS1,TianS2,TianS3,TianS4 so that it does not have to be in different directories
'''
import argparse,os,sys,functools
import numpy as np
import neuroboros as nb
from shared_utils.data_io_utils import parse_range,return_Tian_labels,return_aseg_labels,get_dm,get_rois_and_space
from shared_utils.resource_management_utils import print_memory_usage,save_args_to_json,load_args_from_json,merge_args
from shared_utils.hyperalignment_utils import calculate_connectivity,get_target_ts,calc_dm_across_runs,loo_cv, reliability_weighting_hyperalignment, get_hyperalignment_func
from neuroboros.stats import cronbach_alpha
from glob import glob
from hyperalignment.sparse import initialize_sparse_matrix
from hyperalignment import searchlight_template, compute_template, searchlight_hyperalignment, searchlight_weights
'''
Helper functions
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-json", type=str, default=None,
                       help="Load configuration from JSON file (overrides other arguments)")
    parser.add_argument("--dataset",help="datasets in neuroboros or defined in your io function") #choices=["HBN_full","HBN_sample","Budapest","Raiders"]
    parser.add_argument("--outroot", default = '.', help="Root output directory")  
    parser.add_argument("--extra_str", default = '', help="Some memo to tell me the settings I changed")  
    parser.add_argument("--seeds", default="onavg-ico32",help="the searchlight/rois to align")
    parser.add_argument("--targets",nargs="+",default="response",
                        help="target to calculate connectivity, for cortical it is searchlight means and for subcortical it is roi means") # , choices=["None","response","cortical-lr", "subcortical-lr","cortical-subcortical-lr"]
    parser.add_argument("--task")
    parser.add_argument("--alignfunc", choices=["ridge", "procr","","ridgeCV"], default="")
    parser.add_argument("--ridge_alpha",nargs = "+", default = 1e3, type=float, help = "Regularization strength, used only if the args.align is ridge or ridgeCV")
    parser.add_argument("--run", type=parse_range, default=[1])
    parser.add_argument("--subj", type=parse_range, default=[10,11,12])     
    parser.add_argument("--template_dir", default="",help="Path to the template directory")  
    parser.add_argument("--transform_dir", default="",help="Path to the transforms directory")  
    parser.add_argument("--searchlight_radius",default = 20, type=float, help = "Cortical search lights, think of them as approximately ROIs")
    parser.add_argument("--searchlight_center",default = 'onavg-ico32', help = "Will determine the number of searchlights")
    parser.add_argument("--notweighted",action="store_true",default=False, help = "Whether we do distance based weighting to combine searchlights (stage='template' or 'align')")
    parser.add_argument("--reliability_weighting",action="store_true",default=False, help = "Whether to consider voxel reliability")
    parser.add_argument("--surface_space",default='onavg-ico32')
    parser.add_argument("--volume_space",default='mni-2mm')
    parser.add_argument("--surface_resample",default="1step_pial_overlap")
    parser.add_argument("--volume_resample",default="1step_linear_overlap")
    parser.add_argument("--prep", default="default",help = "Nuissance regression method, default includes no censoring of frames and scrub used the saved mask for censoring") # choices=["default", "scrub","default-gsr", "scrub-gsr"]
    parser.add_argument("--zscore_axis",default="searchlight",choices=["searchlight","fullmatrix"],help="Possibilities to zscore connectivity in stage = template or align]")
    parser.add_argument("--separate_zscore",default=True,help="Separately zscore cortical and subcortical connectivity, only meaningful if the target is cortical-subcortical-lr")
    parser.add_argument("--saved_beta_root", default="/dartfs/rc/lab/H/HaxbyLab/datasets/",help="Path to the saved betas, set to "" if not available")  
    parser.add_argument("--dry-run", action="store_true", default=False, 
                       help="Only save configuration JSON without running the main pipeline")
    
    # Parse CLI arguments first
    cli_args = parser.parse_args()

    # If --load-json is provided, load from JSON and override CLI args
    if cli_args.load_json:
        if not os.path.exists(cli_args.load_json):
            parser.error(f"JSON file not found: {cli_args.load_json}")
        
        print(f"Loading configuration from: {cli_args.load_json}")
        json_args = load_args_from_json(cli_args.load_json)
        
        # Merge JSON args with CLI args (CLI takes precedence for specific fields)
        merged_args = merge_args(json_args, cli_args)
        
        # Create a new Namespace object with merged arguments
        args = argparse.Namespace(**merged_args)
    else:
        args = cli_args

    args.stage = "align"
    if 'saved_beta_root' in args and os.path.exists(args.saved_beta_root):
        args.saved_beta_path = os.path.join(args.saved_beta_root,'saved_confound_betas',args.prep,args.dataset)
    if not isinstance(args.targets,list):
        args.targets = [args.targets]
    if 'transform_dir' in args and os.path.exists(args.transform_dir):
        args.outdir = args.transform_dir
    else:
        seedrois,seedspace = get_rois_and_space(args.seeds,args.surface_space,args.volume_space)
        if seedspace == args.surface_space:
            args.outdir = os.path.join(args.outroot,args.dataset,f'{args.surface_space}_{args.surface_resample}',args.extra_str,'transformation_to_template')
        else:
            args.outdir = os.path.join(args.outroot,args.dataset,f'{args.volume_space}_{args.volume_resample}',args.extra_str,'transformation_to_template')
    
    if not('template_dir' in args and os.path.exists(args.template_dir)):
        if seedspace == args.surface_space:
            args.template_dir = os.path.join(args.outroot,args.dataset,f'{args.surface_space}_{args.surface_resample}','templates')
        else:
            args.template_dir = os.path.join(args.outroot,args.dataset,f'{args.volume_space}_{args.volume_resample}','templates')

    config_path = save_args_to_json(args)
    if args.dry_run:
        print("DRY RUN: Configuration saved. Exiting without running pipeline.")
        print(f"Config file: {config_path}")
        sys.exit(0)

    kwargs = { # for older datasets, there might be multiple versions/folders
    'space': [args.surface_space, args.volume_space],
    'resample': [args.surface_resample,args.volume_resample],
    'prep':args.prep
    }   
    if args.dataset.lower() in ["budapest", "raiders"]:
        dset=nb.datasets.datasets[args.dataset.lower()](fp_version="20.2.7",**kwargs)
    elif args.dataset.lower() in nb.datasets.datasets.keys():
        dset=nb.datasets.datasets[args.dataset.lower()](**kwargs)
    else:
        ValueError("Unsupported Dataset")
    sids = np.array(dset.subjects)[args.subj]

    if args.zscore_axis=="searchlight":
        zscore_ax= 0
    elif args.zscore_axis=="fullmatrix":
        zscore_ax = None
    else:
        raise ValueError("Unsupported zscore_axis")

    os.makedirs(args.outdir, exist_ok=True)    

    # Begin computation
    # Make seed ts
    seedrois,seedspace = get_rois_and_space(args.seeds,args.surface_space,args.volume_space)
    # seed_ts = get_dm(dset, sids, seedrois, args.task, args.run, seedspace,
    #                 atlas=args.seeds,saved_beta_path=args.saved_beta_path)
    for roi in seedrois:
        if roi in ['l','r']:
            tpl_suffix = 'h.npy'
        else:
            tpl_suffix = '.npy'
        # import template, we'll fix the template naming part later!
        if len(args.targets) == 1: 
            if args.targets[0] == 'response':
                tpl_fn = glob(os.path.join(args.template_dir,f'*targets_response_{roi.replace("_subj","")}{tpl_suffix}'))
            elif args.targets[0] == 'onavg-ico8':
                tpl_fn = glob(os.path.join(args.template_dir,f'*targets_cortical-lr_{roi.replace("_subj","")}{tpl_suffix}'))
            else:
                tpl_fn = glob(os.path.join(args.template_dir,f'*targets_subcortical-lr_{roi.replace("_subj","")}{tpl_suffix}'))
        else:
            raise ValueError("I need to figure out how to do the multiple targets later!")
        assert len(tpl_fn)==1, f"Found {len(tpl_fn )} template files but only expecting one!"
        tpl = np.load(tpl_fn[0])

        for sid in sids:
            if roi in ['l','r']:
                sls, dists = nb.sls(roi, args.searchlight_radius, space=args.surface_space, center_space=args.searchlight_center, mask=True, return_dists=True)
                nv = tpl.shape[1]
                mat0 = initialize_sparse_matrix(sls,nv)
                if args.notweighted:
                    weights = searchlight_weights(sls,dists=None,radius=args.searchlight_radius)
                else:
                    weights = searchlight_weights(sls,dists=dists,radius=args.searchlight_radius)
                out_fn = os.path.join(args.outdir, f"{sid}_targets_{args.targets[0]}_{roi}h_{args.alignfunc}.npz")
            else:
                out_fn = os.path.join(args.outdir, f"{sid}_targets_{args.targets[0]}_{roi}_{args.alignfunc}.npy")
            if os.path.exists(out_fn):
                continue
            dm_allruns = calc_dm_across_runs(dset,sid, roi, seedspace, args.targets,args.seeds,args.task,args.run,args.saved_beta_path,zscore_axis=zscore_ax,separate_zscore=args.separate_zscore)
            if args.reliability_weighting:
                calpha = cronbach_alpha(dm_allruns,rep_axis=0,var_axis=1)
            else:
                calpha = None
            
            if args.alignfunc == 'ridgeCV':
                assert len(args.run)>1, "Need more than more run for leave-one-run-out cross validation"
                avg_mse_train = np.zeros((len(args.ridge_alpha)))
                avg_mse_val = np.zeros((len(args.ridge_alpha)))
                for ii,alpha in enumerate(args.ridge_alpha):
                    func = get_hyperalignment_func('ridge',alpha)
                    if roi in ['l','r']:
                        func = functools.partial(searchlight_hyperalignment,sls=sls,mat0=mat0,sl_func = func,weights=weights)
                    avg_mse_train[ii],avg_mse_val[ii] = loo_cv(dm_allruns, tpl, func, voxel_reliability_scores=calpha)
                    best_idx = np.argmin(avg_mse_val)
                    best_alpha = args.ridge_alpha[best_idx] 
            elif args.alignfunc=='ridge':
                assert not isinstance(args.ridge_alpha,list), "not using cross validation, cannot have multiple ridge alphas"  
                best_alpha = args.ridge_alpha
            else:
                best_alpha=None
                
            func = get_hyperalignment_func(args.alignfunc,best_alpha)

            X = np.mean(dm_allruns,axis=0) # for the final R we use the average of all training runs
            if roi in ['l','r']:
                searchlight_func = functools.partial(searchlight_hyperalignment,sls=sls,mat0=mat0,sl_func = func,weights=weights)
                xfm_final = reliability_weighting_hyperalignment(
                        X, tpl, searchlight_func, calpha
                    )
            else:
                xfm_final = reliability_weighting_hyperalignment(
                        X, tpl, func, calpha
                    )
            
            nb.save(out_fn,xfm_final)
                
