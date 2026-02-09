#!/usr/bin/env python
# coding: utf-8
'''
Script to call for making the template.
'''
import argparse,os,sys
import numpy as np
import neuroboros as nb
from shared_utils.data_io_utils import parse_range,return_Tian_labels,return_aseg_labels,get_dm,get_rois_and_space
from shared_utils.resource_management_utils import print_memory_usage,save_args_to_json,load_args_from_json,merge_args
from shared_utils.hyperalignment_utils import calculate_connectivity,get_target_ts
from hyperalignment import searchlight_template, compute_template

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
    parser.add_argument("--run", type=parse_range, nargs="+", default=[1])
    parser.add_argument("--subj", type=parse_range, nargs="+", default=[10,11,12])     
    parser.add_argument("--template_dir", default="",help="Path to the template directory")  
    parser.add_argument("--searchlight_radius",default = 20, type=float, help = "Cortical search lights, think of them as approximately ROIs")
    parser.add_argument("--searchlight_center",default = 'onavg-ico32', help = "Will determine the number of searchlights")
    parser.add_argument("--notweighted",action="store_true",default=False, help = "Whether we do distance based weighting to combine searchlights (stage='template' or 'align')")
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

    args.stage = "template"
    if 'saved_beta_root' in args and os.path.exists(args.saved_beta_root):
        args.saved_beta_path = os.path.join(args.saved_beta_root,'saved_confound_betas',args.prep,args.dataset)
        print(args.saved_beta_path)
    if not isinstance(args.targets,list):
        args.targets = [args.targets]
        
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

    if  args.template_dir:
        template_dir = args.template_dir
    else:
        template_dir = os.path.join(args.outdir, "templates")
    os.path.makedirs(template_dir,exist_ok=True)

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
    os.makedirs(template_dir, exist_ok=True)


    # Begin computation
    # Make seed ts
    seedrois,seedspace = get_rois_and_space(args.seeds,args.surface_space,args.volume_space)
    
    for roi in seedrois:
        seed_ts = get_dm(dset, sids, roi, args.task, args.run, seedspace,
                        atlas=args.seeds,saved_beta_path=args.saved_beta_path)
    
    # Get final dm (response or connectivity)
    def prepare_each_roi(seed_ts,roi,targets):
        allconn = []
        for target in targets:
            if target == "None":
                raise ValueError("Templates not needed for anatomical alignment. Exiting")
            elif target == 'response':
                dm = [seed_ts[sid][roi] for sid in seed_ts.keys()]
            else: # connectivity
                targetrois,targetspace = get_rois_and_space(target,args.surface_space,args.volume_space)
                dm = []
                for sid in sids:
                    if target==args.seeds:
                        target_ts = seed_ts.copy()
                    else:
                        target_ts = get_dm(dset, sid, targetrois, args.task, args.run, targetspace,
                                atlas=target,saved_beta_path=args.saved_beta_path)
                    if target.startswith('onavg'):
                        MAPPINGS = {roi:nb.mapping(roi, args.surface_space, target, mask=True) for roi in targetrois}
                        target_ts = get_target_ts(target_ts,'mapping',target_item=MAPPINGS)[0]
                    else:
                        target_ts = get_target_ts(target_ts,'searchlight_mean')[0]   
                    dm.append(calculate_connectivity(target_ts,seed_ts[sid][roi],zscore_axis = zscore_ax))
            try:
                dm = np.stack(dm,axis=0) 
            except:
                dm = np.concatenate(dm,axis=1) # if they don't have the same vertices
            allconn.append(dm)
        return allconn

    for roi in seedrois:
        if roi in ['l','r']: # Whole-brain template (searchlight)
            tpl_fn = os.path.join(template_dir, f"seeds-{args.seeds}_targets-{'_'.join(args.targets)}_{roi}h.npy")
        else: # Local template
            tpl_fn = os.path.join(template_dir, f"seeds-{args.seeds}_targets-{'_'.join(args.targets)}_{roi}.npy")
        if not os.path.exists(tpl_fn):
            dm = prepare_each_roi(roi,targets)
            if roi in ['l','r']:
                sls, dists = nb.sls(roi, args.searchlight_radius, space=args.surface_space, center_space=args.searchlight_center, mask=True, return_dists=True)
                if not args.notweighted:
                    nb.record(tpl_fn, searchlight_template)(dm, sls, dists,args.searchlight_radius,n_jobs=1,tpl_kind="pca")
                else:
                    nb.record(tpl_fn, searchlight_template)(dm, sls, None,args.searchlight_radius,n_jobs=1,tpl_kind="pca")
            else:
                nb.record(tpl_fn, compute_template)(dm, kind='pca', max_npc=dm.shape[1], common_topography=True)
            print(f"Template saved: {tpl_fn}")
