import argparse
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

import config


def plot_results(args):
    
    # Arugments & parameters
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    print(taxonomy_level)
    
    filename = 'main'
    prefix = ''
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 5000
    data_type = 'validate'
    
    iterations = np.arange(0, max_plot_iteration, 200)
    measure_keys = ['mAP', 'micro_auprc', 'micro_f1', 'macro_auprc']
    
    def _load_stat(model_type):
        validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
            'taxonomy_level={}'.format(taxonomy_level), 
            'holdout_fold={}'.format(holdout_fold), model_type, 
            'validate_statistics.pickle')
        
        statistics_list = cPickle.load(open(validate_statistics_path, 'rb'))
        
        average_precisions = np.array([statistics['average_precision'] \
            for statistics in statistics_list])    # (N, classes_num)
            
        mAP = np.mean(average_precisions, axis=-1)
        
        micro_auprc = np.array([statistics['micro_auprc'] \
            for statistics in statistics_list])
            
        micro_f1 = np.array([statistics['micro_f1'] \
            for statistics in statistics_list])
            
        macro_auprc = np.array([statistics['macro_auprc'] \
            for statistics in statistics_list])
            
        legend = '{}'.format(model_type)
        
        results = {'mAP': mAP, 'micro_auprc': micro_auprc, 'micro_f1': micro_f1, 
            'macro_auprc': macro_auprc, 'legend': legend}
            
        print('Model type: {}'.format(model_type))
        # print('    mAP: {:.3f}'.format(mAP[-1]))
        # print('    micro_auprc: {:.3f}'.format(micro_auprc[-1]))
        # print('    micro_f1: {:.3f}'.format(micro_f1[-1]))
        # print('    macro_auprc: {:.3f}'.format(macro_auprc[-1]))
        idx = micro_auprc.argmax()
        print('    mAP: {:.3f}'.format(mAP[idx]))
        print('    micro_auprc: {:.3f}'.format(micro_auprc[idx]))
        print('    micro_f1: {:.3f}'.format(micro_f1[idx]))
        print('    macro_auprc: {:.3f}'.format(macro_auprc[idx]))
        
        return results
    
    results_dict = {}
    # for model_type in ['Cnn_5layers_AvgPooling', 'Cnn_9layers_AvgPooling', 'Cnn_9layers_MaxPooling', 'Cnn_13layers_AvgPooling', 'Cnn_9layers_AvgPooling_Emb', 'Cnn_9layers_AvgPooling_GCNEmb', 'Cnn_9layers_AvgPooling_GCNEmb_aser_pre_conj', 'Cnn_9layers_AvgPooling_GCNEmb_ontology_aser_pre_conj']:
    for model_type in os.listdir(f"{workspace}/statistics/main/logmel_64frames_64melbins/taxonomy_level=fine/holdout_fold=1"):
        # print(model_type)
        # if not ("ontology_aser" in model_type or model_type == "Cnn_9layers_AvgPooling"):
        #     continue
        # if not ("DoubleGCN" in model_type or model_type == "Cnn_9layers_AvgPooling"):
        #     continue

        try:
            results_dict[model_type] = _load_stat(model_type)
        except Exception as e:
            print(e)
            continue
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    for n, measure_key in enumerate(measure_keys):
        lines = []
        row = n // 2
        col = n % 2

        for model_key in results_dict.keys():
            label = results_dict[model_key]['legend']
            if label == "Cnn_9layers_AvgPooling":
                label = "backbone"
            else:
                label = re.search(r"[^_]+_[^_]+_[^_]+_(.+)", label).group(1)
            line, = axs[n // 2, n % 2].plot(results_dict[model_key][measure_key], label=label)
            lines.append(line)
        
        axs[row, col].set_title(measure_key)    
        axs[row, col].legend(handles=lines, loc=4)
        axs[row, col].set_ylim(0, 1.0)
        axs[row, col].set_xlabel('Iterations')
        axs[row, col].grid(color='b', linestyle='solid', linewidth=0.2)
        # axs[row, col].xaxis.set_ticks(np.arange(0, len(iterations), len(iterations) // 4))
        # axs[row, col].xaxis.set_ticklabels(np.arange(0, max_plot_iteration, max_plot_iteration // 4))
        
    plt.tight_layout()
    plt.savefig(os.path.join(workspace, f"stats_{taxonomy_level}.jpg"))
    # plt.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)

    args = parser.parse_args()
    
    plot_results(args)