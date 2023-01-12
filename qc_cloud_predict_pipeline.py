import pandas as pd
import argparse
import shutil
import os
import subprocess
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm 
import numpy as np
from umap import UMAP
from sklearn import preprocessing, metrics
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import plotly.express as px
import plotly
import joblib
import pickle as pkl
from google.cloud import aiplatform

#local imports
from QC.utils import shell_do, get_common_snps, rm_tmps, merge_genos
from QC.qc import callrate_prune, het_prune, sex_prune, related_prune, variant_prune
from Ancestry.ancestry import plot_3d, get_raw_files, munge_training_data, calculate_pcs, transform, train_umap_classifier, run_admixture, umap_transform_with_fitted, split_cohort_ancestry

from utils.dependencies import check_plink, check_plink2, check_admixture

plink_exec = check_plink()
plink2_exec = check_plink2()
admix_exec = check_admixture()

def load_umap_classifier(pipe_clf, X_test, y_test):
    step = "load_umap_classifier"
    print()
    print(f"RUNNING: {step}")
    print()

    # convert to list (needed for vertex ai predictions)
    X_test_arr = np.array(X_test).tolist()

    # no more score function so get testing balanced accuracy based on vertex ai predictions
    prediction = pipe_clf.predict(instances=X_test_arr)
    pipe_clf_pred = prediction.predictions
    pipe_clf_pred = [int(i) for i in pipe_clf_pred]

    test_acc = metrics.balanced_accuracy_score(y_test, pipe_clf_pred)
    print(f'Balanced Accuracy on Test Set: {test_acc}')

    margin_of_error = 1.96 * np.sqrt((test_acc * (1-test_acc)) / np.shape(y_test)[0])
    print(f"Balanced Accuracy on Test Set, 95% Confidence Interval: ({test_acc-margin_of_error}, {test_acc+margin_of_error})")

    # confustion matrix
    pipe_clf_c_matrix = metrics.confusion_matrix(y_test, pipe_clf_pred)

    out_dict = {
        'classifier': pipe_clf,
        'confusion_matrix': pipe_clf_c_matrix,
        'test_accuracy': test_acc
    }

    return out_dict


def predict_ancestry_from_pcs(projected, pipe_clf, label_encoder, out):
    step = "predict_ancestry"
    print()
    print(f"RUNNING: {step}")
    print()

    le = label_encoder

    # set new samples aside for labeling after training the model
    X_new = projected.drop(columns=['FID','IID','label'], axis=1)

    # convert to numpy array
    X_new_arr = np.array(X_new)

    # if num samples > ~2000, need to split into multiple batches of predictions
    num_splits = round((X_new.shape[0] / 2000), 0)

    y_pred = []

    if num_splits > 0:
        for arr in np.array_split(X_new_arr, num_splits):
            # convert to list (needed for vertex ai predictions)
            arr = arr.tolist()
            # get predictions from vertex ai
            prediction = pipe_clf.predict(instances=arr)
            pred = prediction.predictions
            pred = [int(i) for i in pred]
            y_pred += pred
    else:
         # convert to list (needed for vertex ai predictions)
        arr = X_new_arr.tolist()
        # get predictions from vertex ai
        prediction = pipe_clf.predict(instances=arr)
        pred = prediction.predictions
        pred = [int(i) for i in pred]
        y_pred += pred

    ancestry_pred = le.inverse_transform(y_pred)
    projected.loc[:,'label'] = ancestry_pred

    print()
    print('predicted:\n', projected.label.value_counts())
    print()

    projected[['FID','IID','label']].to_csv(f'{out}_umap_linearsvc_predicted_labels.txt', sep='\t', index=False)

    data_out = {
        'ids': projected.loc[:,['FID','IID','label']],
        'X_new': X_new,
        'y_pred': ancestry_pred,
        'label_encoder': le
    }

    outfiles_dict = {
        'labels_outpath': f'{out}_umap_linearsvc_predicted_labels.txt'
    }

    out_dict = {
        'data': data_out,
        'metrics': projected.label.value_counts(),
        'output': outfiles_dict
    }

    return out_dict

def run_ancestry(geno_path, out_path, ref_panel, ref_labels, train=False, train_param_grid=None):
    step = "predict_ancestry"
    print()
    print(f"RUNNING: {step}")
    print()
    
    print(os.getcwd())
    outdir = os.path.dirname(out_path)
    plot_dir = f'{outdir}/plot_ancestry'

    # create directories if not already in existence
    # os.makedirs(plot_dir, exist_ok=True)

    raw = get_raw_files(
        geno_path=geno_path,
        ref_path=ref_panel,
        labels_path=ref_labels,
        out_path=out_path,
        train=train
    )

    train_split = munge_training_data(labeled_ref_raw=raw['raw_ref'])

    calc_pcs = calculate_pcs(
        X_train=train_split['X_train'], 
        X_test=train_split['X_test'],
        y_train=train_split['y_train'],
        y_test=train_split['y_test'],
        train_ids=train_split['train_ids'],
        test_ids=train_split['test_ids'],
        raw_geno=raw['raw_geno'],
        label_encoder=train_split['label_encoder'],
        out=out_path,
        plot_dir=plot_dir
    )

    # initialize connection to vertex ai model endpoint
    cloud_project = 'genotools'
    cloud_region = 'us-central1'
    cloud_endpoint_id = '9131351709702946816'

    aiplatform.init(project=cloud_project, location=cloud_region)
    endpoint = aiplatform.Endpoint(cloud_endpoint_id)

    # if not training, pass the endpoint instead of model
    if not train:
        trained_clf = load_umap_classifier(
            pipe_clf=endpoint,
            X_test=calc_pcs['X_test'],
            y_test=train_split['y_test']
        )

    # otherwise, train a new model
    else:
        trained_clf = train_umap_classifier(
            X_train=calc_pcs['X_train'],
            X_test=calc_pcs['X_test'],
            y_train=train_split['y_train'],
            y_test=train_split['y_test'],
            label_encoder=train_split['label_encoder'],
            out=out_path,
            plot_dir=plot_dir
        )

    pred = predict_ancestry_from_pcs(
        projected=calc_pcs['new_samples_projected'],
        pipe_clf=trained_clf['classifier'],
        label_encoder=train_split['label_encoder'],
        out=out_path
    )

    admix = run_admixture(
        predicted_labels=pred['data']['ids'],
        train_pca=calc_pcs['labeled_train_pca'],
        out_path=out_path
    )

    umap_transforms = umap_transform_with_fitted(
        ref_pca=calc_pcs['labeled_ref_pca'],
        X_new=pred['data']['X_new'],
        y_pred=admix['data']['ids']
    )
    
#     x_min, x_max = min(umap_transforms['total_umap'].iloc[:,0]), max(umap_transforms['total_umap'].iloc[:,0])
#     y_min, y_max = min(umap_transforms['total_umap'].iloc[:,1]), max(umap_transforms['total_umap'].iloc[:,1])
#     z_min, z_max = min(umap_transforms['total_umap'].iloc[:,2]), max(umap_transforms['total_umap'].iloc[:,2])

#     x_range = [x_min-5, x_max+5]
#     y_range = [y_min-5, y_max+5]
#     z_range = [z_min-5, z_max+5]

#     plot_3d(
#         umap_transforms['total_umap'],
#         color='label',
#         symbol='dataset',
#         plot_out=f'{plot_dir}/plot_total_umap',
#         title='UMAP of New and Reference Samples',
#         x=0,
#         y=1,
#         z=2,
#         x_range=x_range,
#         y_range=y_range,
#         z_range=z_range
#     )

#     plot_3d(
#         umap_transforms['ref_umap'],
#         color='label',
#         symbol='dataset',
#         plot_out=f'{plot_dir}/plot_ref_umap',
#         title="UMAP of Reference Samples",
#         x=0,
#         y=1,
#         z=2,
#         x_range=x_range,
#         y_range=y_range,
#         z_range=z_range
#     )

#     plot_3d(
#         umap_transforms['new_samples_umap'],
#         color='label',
#         symbol='dataset',
#         plot_out=f'{plot_dir}/plot_predicted_samples_umap',
#         title='UMAP of New Samples',
#         x=0,
#         y=1,
#         z=2,
#         x_range=x_range,
#         y_range=y_range,
#         z_range=z_range
#     )

    # return more stuff as needed but for now, just need predicted labels, predicted labels out path, and predicted counts
    data_dict = {
        'predict_data': pred['data'],
        'confusion_matrix': trained_clf['confusion_matrix'],
        'train_pcs': calc_pcs['labeled_train_pca'],
        'ref_pcs': calc_pcs['labeled_ref_pca'],
        'projected_pcs': calc_pcs['new_samples_projected'],
        'admix_data': admix['data'],
        'total_umap': umap_transforms['total_umap'],
        'ref_umap': umap_transforms['ref_umap'],
        'new_samples_umap': umap_transforms['new_samples_umap'],
        'label_encoder': train_split['label_encoder']
    }

    metrics_dict = {
        'predicted_counts': pred['metrics'],
        'adjusted_counts': admix['metrics'],
        'test_accuracy': trained_clf['test_accuracy']
    }

    outfiles_dict = {
        'predicted_labels': pred['output'],
        'adjusted_labels': admix['output']
    }
    
    out_dict = {
        'step': step,
        'data': data_dict,
        'metrics': metrics_dict,
        'output': outfiles_dict
    }

    return out_dict


# cloud predict pipeline
parser = argparse.ArgumentParser(description='Arguments for Genotyping QC (data in Plink .bim/.bam/.fam format)')
parser.add_argument('--geno', type=str, default='nope', help='Genotype: (string file path). Path to PLINK format genotype file, everything before the *.bed/bim/fam [default: nope].')
parser.add_argument('--ref', type=str, default='nope', help='Genotype: (string file path). Path to PLINK format reference genotype file, everything before the *.bed/bim/fam.')
parser.add_argument('--ref_labels', type=str, default='nope', help='tab-separated plink-style IDs with ancestry label (FID  IID label) with no header')
parser.add_argument('--train', type=bool, default=False, help='Whether to train a new model or use pretrained model for prediction')
parser.add_argument('--callrate', type=float, default=0.02, help='Minimum Callrate threshold for QC')
parser.add_argument('--out', type=str, default='nope', help='Prefix for output (including path)')

args = parser.parse_args()

geno_path = args.geno
ref_panel = args.ref
ref_labels = args.ref_labels
train = args.train
callrate = args.callrate
out_path = args.out

# sample-level pruning and metrics
callrate_out = f'{geno_path}_callrate'
callrate = callrate_prune(geno_path, callrate_out, mind=callrate)

sex_out = f'{callrate_out}_sex'
sex = sex_prune(callrate_out, sex_out)


# run ancestry methods
ancestry_out = f'{sex_out}_ancestry'
# no longer pass a model path, instead specify if training or not in pipeline (defaults to train=False)
ancestry = run_ancestry(geno_path=sex_out, out_path=ancestry_out, ref_panel=ref_panel, ref_labels=ref_labels, train=train)

# get ancestry counts to add to output .h5 later
ancestry_counts_df = pd.DataFrame(ancestry['metrics']['adjusted_counts']).reset_index()
ancestry_counts_df.columns = ['label', 'count']


# split cohort into individual ancestry groups
pred_labels_path = ancestry['output']['adjusted_labels']['labels_outpath']
cohort_split = split_cohort_ancestry(geno_path=sex_out, labels_path=pred_labels_path, out_path=ancestry_out)

# ancestry-specific pruning steps
het_dict = dict()
related_dict = dict()
variant_dict = dict()

for geno, label in zip(cohort_split['paths'], cohort_split['labels']):

    # related
    related_out = f'{geno}_related'
    related = related_prune(geno, related_out, prune_related=False)
    related_dict[label] = related
    
    # het
    het_out = f'{related_out}_het'
    het = het_prune(related_out, het_out)
    het_dict[label] = het
    
    # variant
    variant_out = f'{het_out}_variant'
    if het['pass']:
        variant = variant_prune(het_out, variant_out)
        variant_dict[label] = variant
    else:
        variant = variant_prune(related_out, variant_out)
        variant_dict[label] = variant



# copy output to out_path
for label, data in variant_dict.items():
    if data['pass']:
        for suffix in ['bed','bim','fam','log']:
            plink_file = f"{data['output']['plink_out']}.{suffix}"
            plink_outfile = f'{out_path}_{label}.{suffix}'
            shutil.copyfile(src=plink_file, dst=plink_outfile)

# copy list of related samples to out_path
for label, data in related_dict.items():
    if data['pass']:
        related_file = f"{data['output']['related_samples']}"
        related_outfile = f"{out_path}_{label}.related"
        shutil.copyfile(src=related_file, dst=related_outfile)

# build report- eventually make this an individual method
steps = [callrate, sex]
steps2 = [het_dict, related_dict, variant_dict]
metrics_df = pd.DataFrame()
pruned_samples_df = pd.DataFrame()

for item in steps:
    
    step = item['step']
    pf = item['pass']
    level = 'sample'
    ancestry_label = 'all'
    
    for metric, value in item['metrics'].items():
        tmp_metrics_df = pd.DataFrame({'step':[step], 'pruned_count':[value], 'metric':[metric], 'ancestry':[ancestry_label], 'level':[level], 'pass': [pf]})
        metrics_df = metrics_df.append(tmp_metrics_df)
    
    samplefile = item['output']['pruned_samples']
    if os.path.isfile(samplefile):
        pruned = pd.read_csv(samplefile, sep='\t')
        if pruned.shape[0] > 0:
            pruned.loc[:,'step'] = step
            pruned_samples_df = pruned_samples_df.append(pruned[['FID','IID','step']])
        
for item in steps2:
    for ancestry_label, metrics in item.items():
        
        step = metrics['step']
        pf = metrics['pass']
        
        if step in ['het_prune','related_prune']:
            level = 'sample'

            samplefile = metrics['output']['pruned_samples']
            if os.path.isfile(samplefile):
                pruned = pd.read_csv(samplefile, sep='\t', header=0, usecols=[0,1], names=['FID','IID'])
                if pruned.shape[0] > 0:
                    pruned.loc[:,'step'] = step
                    pruned_samples_df = pruned_samples_df.append(pruned[['FID','IID','step']])
            
        else:
            level = 'variant'

        for metric, value in metrics['metrics'].items():
            tmp_metrics_df = pd.DataFrame({'step':[step], 'pruned_count':[value], 'metric':[metric], 'ancestry':[ancestry_label], 'level':[level], 'pass': [pf]})
            metrics_df = metrics_df.append(tmp_metrics_df)

metrics_df.reset_index(drop=True, inplace=True)


# build output hdf
metrics_outfile = f'{out_path}.QC.metrics.h5'

le = ancestry['data']['label_encoder']
confusion_matrix = ancestry['data']['confusion_matrix']
conf_mat_df = pd.DataFrame(confusion_matrix)
conf_mat_df.columns = le.inverse_transform([i for i in range(10)])
conf_mat_df.index = le.inverse_transform([i for i in range(10)])

ref_pcs = ancestry['data']['ref_pcs']
projected_pcs = ancestry['data']['projected_pcs']
total_umap = ancestry['data']['total_umap']
ref_umap = ancestry['data']['ref_umap']
new_samples_umap = ancestry['data']['new_samples_umap']
pred_ancestry_labels = ancestry['data']['admix_data']['ids']

metrics_df.to_hdf(metrics_outfile, key='QC', mode='w')
pruned_samples_df.to_hdf(metrics_outfile, key='pruned_samples')
ancestry_counts_df.to_hdf(metrics_outfile, key='ancestry_counts')
pred_ancestry_labels.to_hdf(metrics_outfile, key='ancestry_labels')
conf_mat_df.to_hdf(metrics_outfile, key='confusion_matrix', index=True)
ref_pcs.to_hdf(metrics_outfile, key='ref_pcs')
projected_pcs.to_hdf(metrics_outfile, key='projected_pcs')
total_umap.to_hdf(metrics_outfile, key='total_umap')
ref_umap.to_hdf(metrics_outfile, key='ref_umap')
new_samples_umap.to_hdf(metrics_outfile, key='new_samples_umap')