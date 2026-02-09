#!/usr/bin/env python
# coding: utf-8
import argparse, os, sys
import numpy as np
import neuroboros as nb
from joblib import Parallel, delayed
from shared_utils.data_io_utils import parse_range,get_rois_and_space
from shared_utils.resource_management_utils import save_args_to_json,load_args_from_json,merge_args
from shared_utils.hyperalignment_utils import calc_dm_across_runs
from neuroboros.stats import cronbach_alpha
'''
Helper functions
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-json", type=str, default=None,
                       help="Load configuration from JSON file (overrides other arguments)")
    parser.add_argument("--dataset",help="datasets in neuroboros or defined in your io function") #choices=["HBN_full","HBN_sample","Budapest","Raiders"]
    parser.add_argument("--outroot", default='.', help="Root output directory")  
    parser.add_argument("--extra_str", default='', help="Some memo to tell me the settings I changed")  
    parser.add_argument("--seeds", default="onavg-ico32",help="the searchlight/rois to align")
    parser.add_argument("--targets",nargs="+",default="response",
                        help="target to calculate connectivity, for cortical it is searchlight means and for subcortical it is roi means") # , choices=["None","response","cortical-lr", "subcortical-lr","cortical-subcortical-lr"]
    parser.add_argument("--task")
    parser.add_argument("--run", type=parse_range, nargs="+", default=[1])
    parser.add_argument("--subj", type=parse_range, nargs="+", default=[10,11,12])     
    parser.add_argument("--template_dir", default="",help="Path to the template directory")  
    parser.add_argument("--surface_space",default='onavg-ico32')
    parser.add_argument("--volume_space",default='mni-2mm')
    parser.add_argument("--surface_resample",default="1step_pial_overlap")
    parser.add_argument("--volume_resample",default="1step_linear_overlap")
    parser.add_argument("--prep", default="default",help = "Nuissance regression method, default includes no censoring of frames and scrub used the saved mask for censoring") # choices=["default", "scrub","default-gsr", "scrub-gsr"]
    parser.add_argument("--zscore_axis",default="searchlight",choices=["searchlight","fullmatrix"],help="Possibilities to zscore connectivity in stage = template or align]")
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

    args.stage = "connectivity_reliability"
    if 'saved_beta_root' in args and os.path.exists(args.saved_beta_root):
        args.saved_beta_path = os.path.join(args.saved_beta_root,'saved_confound_betas',args.prep,args.dataset)
    else:
        args.saved_beta_path = None

    seedrois,seedspace = get_rois_and_space(args.seeds,args.surface_space,args.volume_space)
    if seedspace == args.surface_space:
        args.outdir = os.path.join(args.outroot,args.dataset,f'{args.surface_space}_{args.surface_resample}',args.extra_str,'connectivity_reliability')
    else:
        args.outdir = os.path.join(args.outroot,args.dataset,f'{args.volume_space}_{args.volume_resample}',args.extra_str,'connectivity_reliability')
    
    config_path = save_args_to_json(args)
    if args.dry_run:
        print("DRY RUN: Configuration saved. Exiting without running pipeline.")
        print(f"Config file: {config_path}")
        sys.exit(0)
    
    dataset= args.dataset #'Budapest'
    prep = args.prep

    if args.zscore_axis=="searchlight":
        zscore_ax= 0
    elif args.zscore_axis=="fullmatrix":
        zscore_ax = None
    else:
        raise ValueError("Unsupported zscore_axis")
    
    os.makedirs(args.outdir, exist_ok=True)    

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

    if args.seeds.startswith('Tian'):
        seedrois = ['Tian_Subcortex']
    for sid in sids:
        for target in args.targets:
            dm_allruns = calc_dm_across_runs(dset,sid, seedrois, seedspace, target,args.seeds,args.task,args.run,args.saved_beta_path,zscore_axis=zscore_ax)
            calpha = cronbach_alpha(dm_allruns,rep_axis=0,var_axis=1)
            save_cfn = f"{args.outdir}/{sid}_CronbachAlpha_run{''.join(map(str,args.run))}_reliability_seeds-{args.seeds}_targets-{target}.npy"
            np.save(save_cfn,calpha)
    
    # for sid in sids:
    #     calc_reliability_single_subj(dset,sid, args.targets,args.seeds,args.task,args.run,args.outdir,args.saved_beta_path)
    # # parallelize for subject
    # Parallel(n_jobs=4)(
    #     delayed(calc_reliability_single_subj)(dset,sid, args.targets,args.seeds,args.task,args.run,args.outdir,args.saved_beta_path,zscore_ax)
    #     for sid in sids
    # )