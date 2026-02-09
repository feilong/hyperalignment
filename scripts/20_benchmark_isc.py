#!/usr/bin/env python
# coding: utf-8
'''
Script to call for getting the isc.
To-do:
- Make it work
- Add memory-efficient way as default: cumulatively add the matrix first

'''
import os,argparse
from shared_utils.data_io_utils import parse_range,return_Tian_labels,return_aseg_labels,get_dm,get_rois_and_space
from shared_utils.resource_management_utils import print_memory_usage,save_args_to_json,load_args_from_json,merge_args
import numpy as np

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
    parser.add_argument("--run", type=parse_range, default=[1])
    parser.add_argument("--subj", type=parse_range, default=[10,11,12])     
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

    args.stage = "isc"

    if 'saved_beta_root' in args and os.path.exists(args.saved_beta_root):
        args.saved_beta_path = os.path.join(args.saved_beta_root,'saved_confound_betas',args.prep,args.dataset)
    if not isinstance(args.targets,list):
        args.targets = [args.targets]
    if not('outdir' in args and os.path.exists(args.outdir)):
        seedrois,seedspace = get_rois_and_space(args.seeds,args.surface_space,args.volume_space)
        if seedspace == args.surface_space:
            args.outdir = os.path.join(args.outroot,args.dataset,f'{args.surface_space}_{args.surface_resample}',args.extra_str,'ISC_response')
        else:
            args.outdir = os.path.join(args.outroot,args.dataset,f'{args.volume_space}_{args.volume_resample}',args.extra_str,'ISC_connectivity')
    
    # calculate dense connectivity intersubject correlation
    isc_resp_dir = os.path.join(args.outroot, "ISC_response")
    isc_conn_dir = os.path.join(args.outroot, "ISC_connectivity")
    os.makedirs(isc_resp_dir, exist_ok=True)
    os.makedirs(isc_conn_dir, exist_ok=True)
    
    xfm_key = args.target_flag
    if xfm_key == 'AA':
        resp_out_fn = os.path.join(isc_resp_dir, f"ISC_{args.whichrois}_{args.task}_{xfm_key}.npy")
        conn_out_fn = os.path.join(isc_conn_dir, f"ISC_{args.whichrois}_{args.task}_{xfm_key}.npy")
        between_out_fn = conn_out_fn.replace(f'ISC_{args.whichrois}','ISC_betweencorticalsubcortical')
    else:
        resp_out_fn = os.path.join(isc_resp_dir, f"ISC_{args.whichrois}_{args.task}_{xfm_key}_{args.align}.npy")
        conn_out_fn = os.path.join(isc_conn_dir, f"ISC_{args.whichrois}_{args.task}_{xfm_key}_{args.align}.npy")
        between_out_fn = conn_out_fn.replace(f'ISC_{args.whichrois}','ISC_betweencorticalsubcortical')
    
    aligned_ts = {}
    for sid in sids[subj]:
        if xfm_key == 'AA':
            aligned_ts[sid] = np.concatenate([all_ts[sid][roi] for roi in rois],axis=1).astype(np.float32)
        else:
            xfms = load_saved_transforms(sid, transform_dir,rois,args.target_flag,args.align)
            aligned_ts[sid] = np.concatenate([all_ts[sid][roi] @ xfms[roi] for roi in rois],axis=1).astype(np.float32)
            del xfms   