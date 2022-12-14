import os
import shutil
import pandas as pd
from utils import pretty_model_name

# experiment path
path = 'experiments/convergence'

# rename dictionary
rename_dict = dict()


# dummy model object
class Model(object):
    def __init__(self, mdl_name, mdl_class):
        self.name = mdl_name.replace(mdl_class, '')


# walk the experimental directory
for path, subdir, filenames in os.walk(path, topdown=False):

    # loop over model classes
    for model_class in ['Normal', 'Student', 'DeepEnsemble', 'MonteCarloDropout']:

        # rename any matching directories
        for target_dir in rename_dict.keys():
            old_dir = target_dir + model_class
            new_dir = rename_dict[target_dir] + model_class
            if old_dir in subdir:
                print(os.path.join(path, old_dir))
                print(os.path.join(path, new_dir))
                shutil.move(os.path.join(path, old_dir), os.path.join(path, new_dir))

            for file in ['measurements.pkl', 'optimization_history.pkl', 'mean.pkl', 'shap.pkl']:
                if file in filenames:
                    df = pd.read_pickle(os.path.join(path, file))
                    df = df.reset_index('Model')
                    old_name = pretty_model_name(Model(old_dir, model_class), dict())
                    new_name = pretty_model_name(Model(new_dir, model_class), dict())
                    df.loc[df.Model == old_name, 'Model'] = new_name
                    df = df.set_index('Model', append=True)
                    df = df.reorder_levels([df.index.nlevels - 1] + list(range(df.index.nlevels - 1)))
                    df.to_pickle(os.path.join(path, file))
