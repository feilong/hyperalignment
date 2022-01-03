import os
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import zscore

DIR = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.join(DIR, 'data')


class MovieDataset(object):
    def __init__(self, dset_name, flavor, version, surf_type='fsaverage', masked=False):
        self.name = dset_name
        self.flavor = flavor
        self.version = version
        self.surf_type = surf_type
        self.masked = masked

        self.data_dir = f'{DATA_ROOT}/{dset_name}_{version}/preprocessed_correct/{flavor}'
        self.sid_tasks = {}
        for fn in glob(f'{self.data_dir}/*.npy'):
            parts = os.path.basename(fn)[:-4].split('_')
            if len(parts) == 4:
                sid, lr, task, run = parts
            elif len(parts) == 5:
                sid, lr, task, ses, run = parts
            if sid not in self.sid_tasks:
                self.sid_tasks[sid] = set()
            self.sid_tasks[sid].add(task)
        self.all_subjects = sorted(self.sid_tasks.keys())
        self.subject_sets = {
            'all': self.all_subjects,
        }
        if not self.masked:
            if self.surf_type == 'fsaverage':
                self.masks = [np.load(os.path.join(DIR, 'fsaverage_lh_mask.npy')),
                            np.load(os.path.join(DIR, 'fsaverage_rh_mask.npy'))]
            elif self.surf_type == 'fsaverage5':
                self.masks = [np.load(os.path.join(DIR, 'fsaverage5_lh_mask.npy')),
                            np.load(os.path.join(DIR, 'fsaverage5_rh_mask.npy'))]

    def load_data(self, sid, lr, task, run, chop=True, z=True, mask=True):
        fn = f'{self.data_dir}/{sid}_{lr}h_{task}_{run:02d}.npy'
        data = np.load(fn)
        if mask:
            if not self.masked:
                m = self.masks['lr'.index(lr)][:data.shape[1]]
                data = data[:, m]
        else:
            if self.masked:
                raise ValueError
        if z:
            data = np.nan_to_num(zscore(data, axis=0))
        return data

    def load_subj_movie_data(self, sid, lr, runs=None, chop=True, z=True, mask=True, return_lengths=False):
        ds = []
        if runs is None:
            for task_name, task_runs in self.task_info:
                if task_name == self.movie_task:
                    runs = task_runs
                    break
        for run in runs:
            d = self.load_data(sid, lr, self.movie_task, run, chop=chop, z=z, mask=mask)
            ds.append(d)
        lengths = [d.shape[0] for d in ds]
        ds = np.concatenate(ds, axis=0)
        if return_lengths:
            return ds, lengths
        return ds

    def load_movie_data(self, lr, runs, chop=True, z=True, mask=True, group='all'):
        dss = []
        for sid in self.subject_sets[group]:
            dss.append(self.load_subj_movie_data(sid, lr, runs, chop=chop, z=z, mask=mask))
        dss = np.stack(dss, axis=0)
        return dss

    # def load_subj_all_data(self, sid, lr, chop=True, z=True, mask=True):
    #     ds = []
    #     lengths = []
    #     for task, runs in self.task_info:
    #         for run in runs:
    #             d = self.load_data(sid, lr, task, run, chop=chop, z=z, mask=mask)
    #             lengths.append(d.shape[0])
    #             ds.append(d)
    #     ds = np.concatenate(ds, axis=0)
    #     lengths = np.array(lengths)
    #     return ds, lengths


# To use the old 1.4.1 version of the dataset:
# HyperfaceDataset(version='1-4-1', masked=True)
class HyperfaceDataset(MovieDataset):
    def __init__(self, flavor='fmriprep_global-mc-reg_renamed', version='20-1-1', surf_type='fsaverage', masked=False):
        super().__init__(dset_name='hyperface', flavor=flavor, version=version, surf_type=surf_type, masked=masked)
        self.movie_task = 'budapest'
        self.task_info = [
            ['budapest', (1, 2, 3, 4, 5)],
            ['localizer', (1, 2, 3, 4)],
        ]


class SiemensRaidersDataset(MovieDataset):
    def __init__(self, flavor='fmriprep_global-mc-reg', version='20-2-0', surf_type='fsaverage', masked=True):
        super().__init__(dset_name='siemens-raiders', flavor=flavor, version=version, surf_type=surf_type, masked=masked)
        self.movie_task = 'movie'
        self.task_info = [
            ['movie', (1, 2, 3, 4)],
            ['actions', (1, 2, 3, 4, 5, 6, 7, 8)],
        ]
        # 8 + 8 + 7 subjects, 7 + 7 + 6 with localizers
        self.subject_sets['fold1_1'] = ['sid000005', 'sid000007', 'sid000009', 'sid000010', 'sid000012', 'sid000013', 'sid000020', 'sid000021']
        self.subject_sets['fold1_2'] = ['sid000024', 'sid000029', 'sid000034', 'sid000052', 'sid000102', 'sid000114', 'sid000120', 'sid000134', 'sid000142', 'sid000278', 'sid000416', 'sid000433', 'sid000499', 'sid000522', 'sid000535']
        self.subject_sets['fold2_1'] = ['sid000024', 'sid000029', 'sid000034', 'sid000052', 'sid000102', 'sid000114', 'sid000120', 'sid000134']
        self.subject_sets['fold2_2'] = ['sid000005', 'sid000007', 'sid000009', 'sid000010', 'sid000012', 'sid000013', 'sid000020', 'sid000021', 'sid000142', 'sid000278', 'sid000416', 'sid000433', 'sid000499', 'sid000522', 'sid000535']
        self.subject_sets['fold3_1'] = ['sid000142', 'sid000278', 'sid000416', 'sid000433', 'sid000499', 'sid000522', 'sid000535']
        self.subject_sets['fold3_2'] = ['sid000005', 'sid000007', 'sid000009', 'sid000010', 'sid000012', 'sid000013', 'sid000020', 'sid000021', 'sid000024', 'sid000029', 'sid000034', 'sid000052', 'sid000102', 'sid000114', 'sid000120', 'sid000134']

    def load_data(self, sid, lr, task, run, chop=True, z=True, mask=True):
        fn = f'{self.data_dir}/{sid}_{lr}h_{task}_{run:02d}.npy'
        data = np.load(fn)
        if mask:
            if not self.masked:
                m = self.masks['lr'.index(lr)][:data.shape[1]]
                data = data[:, m]
        else:
            if self.masked:
                raise ValueError
        if chop:
            if task == 'movie':
                if run == 1:
                    data = data[:-10]
                elif run == 4:
                    data = data[10:]
                else:
                    data = data[10:-10]
        if z:
            data = np.nan_to_num(zscore(data, axis=0))
        return data


class ForrestDataset(MovieDataset):
    def __init__(self, flavor='fmriprep_global-mc-reg', version='20-1-1', surf_type='fsaverage'):
        super().__init__(dset_name='forrest', flavor=flavor, version=version, surf_type=surf_type)
        self.movie_task = 'movie'
        self.task_info = [
            ['movie', (1, 2, 3, 4, 5, 6, 7, 8)],
            ['objectcategories', (1, 2, 3, 4)],
            ['retmapccw', (1, )],
            ['retmapclw', (1, )],
            ['retmapcon', (1, )],
            ['retmapexp', (1, )],
        ]
        self.subject_sets['fold1_1'] = ['01', '02', '03', '04', '05']
        self.subject_sets['fold1_2'] = ['06', '09', '10', '14', '15', '16', '17', '18', '19', '20']
        self.subject_sets['fold2_1'] = ['06', '09', '10', '14', '15']
        self.subject_sets['fold2_2'] = ['01', '02', '03', '04', '05', '16', '17', '18', '19', '20']
        self.subject_sets['fold3_1'] = ['16', '17', '18', '19', '20']
        self.subject_sets['fold3_2'] = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15']


class ID1000Dataset(MovieDataset):
    def __init__(self, flavor='fmriprep_global-mc-reg', version='1-4-1', surf_type='fsaverage5', masked=True):
        super().__init__(dset_name='id1000', flavor=flavor, version=version, surf_type=surf_type, masked=masked)
        self.movie_task = 'movie'
        self.task_info = [
            ['movie', (1, )],
        ]

    def get_df(self, sids=None):
        df_fn = f'{DIR}/id1000.tsv'  # originally `ds003097/participants.tsv`
        df = pd.read_csv(df_fn, index_col='participant_id', sep='\t')
        if sids is not None:
            df = df.loc[[f'sub-{_}' for _ in sids]]
        return df


class CamCANDataset(MovieDataset):
    def __init__(self, flavor='fmriprep_global-mc-reg', version='20-2-0', surf_type='fsaverage', masked=True):
        super().__init__(dset_name='camcan', flavor=flavor, version=version, surf_type=surf_type, masked=masked)
        self.movie_task = 'movie'
        self.task_info = [
            ['movie', (1, )],
        ]
        for name in ['half1', 'half2']:
            with open(f'{DIR}/camcan_{name}.txt', 'r') as f:
                sids = f.read().splitlines()
            self.subject_sets[name] = sids

    def get_df(self, sids=None):
        df_fn = f'{DIR}/camcan.pkl'
        df = pd.read_pickle(df_fn)
        if sids is not None:
            df = df.loc[sids]
        # df['Sex'], uniques = pd.factorize(df['Sex'], sort=True)
        # print(uniques)
        df['Sex'] = df['Sex'].map({'FEMALE': -1, 'MALE': 1})
        return df


class RaidersDataset(MovieDataset):
    def __init__(self, flavor='fmriprep_global-mc-reg', version='20-1-1', surf_type='fsaverage'):
        super().__init__(dset_name='raiders', flavor=flavor, version=version, surf_type=surf_type)
        self.movie_task = 'raiders'
        self.task_info = [
            ['raiders', (1, 2, 3, 4, 5, 6, 7, 8)],
            # TODO
        ]
        self.subject_sets['all'] = [sid for sid in self.all_subjects if 'raiders' in self.sid_tasks[sid]]
        self.subject_sets['8ch'] = [
            'rid000005', 'rid000011', 'rid000014', 'rid000015', 'rid000020',
            'rid000028', 'rid000029', 'rid000033', 'rid000038', 'rid000042',
            'rid000043']
        self.subject_sets['32ch'] = [
            'rid000001', 'rid000008', 'rid000009', 'rid000012', 'rid000013',
            'rid000016', 'rid000017', 'rid000018', 'rid000019', 'rid000021',
            'rid000022', 'rid000024', 'rid000025', 'rid000026', 'rid000027',
            'rid000031', 'rid000032', 'rid000036', 'rid000037', 'rid000041']

    def load_data(self, sid, lr, task, run, chop=True, z=True, mask=True):
        fn = f'{self.data_dir}/{sid}_{lr}h_{task}_{run:02d}.npy'
        data = np.load(fn)
        if mask:
            m = self.masks['lr'.index(lr)][:data.shape[1]]
            data = data[:, m]
        if chop:
            if task == 'raiders' and run != 1:
                data = data[8:]
            elif task == 'retmapping':
                data = data[10:90]
        if z:
            data = np.nan_to_num(zscore(data, axis=0))
        return data

    def load_subj_all_data(self):
        raise NotImplementedError


class WhiplashDataset(MovieDataset):
    def __init__(self, flavor='fmriprep_global-mc-reg', version='20-0-3', surf_type='fsaverage'):
        super().__init__(dset_name='whiplash', flavor=flavor, version=version, surf_type=surf_type)
        self.movie_task = 'whiplash'
        self.task_info = [
            ['whiplash', (1, 2)],
        ]
        self.subject_sets['all'] = [sid for sid in self.all_subjects if 'whiplash' in self.sid_tasks[sid]]
