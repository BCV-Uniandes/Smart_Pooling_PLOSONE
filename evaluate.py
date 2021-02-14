import os
import h2o 
import pickle
import argparse
import numpy as np
from main import main
from Graph_results import plot_results


parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, default='experiments/best_model',
                    help='Output directory')
parser.add_argument('--port', type=int, default=22222,
                    help='Port to connect h2o')
parser.add_argument('--eval', action='store_true',
                    help='Use this flag to evaluate a model')
parser.add_argument('--path-to-best', type=str, default=None,
                    help='Path to load best model, available when --eval flag is used.')
parser.add_argument('--path-to-val', type=str, default=None,
                    help='Path to load val model, available when --eval flag is used.')
parser.add_argument('--geoinfo', type=str,
                    default='data/GeoInfo.csv')
parser.add_argument('--excelinfo', type=str,
                    default='data/TestCenter.xlsx')
parser.add_argument('--filterdate', type=str,
                    default='2020-05-08')
parser.add_argument('--savegraph', action='store_true',
                    help='save individual graphs of experiments')
parser.add_argument('--poolsize', type=int, default=10, help='poolsize')

args = parser.parse_args()

h2o.init(port=args.port)

output_path = args.output_dir

vec_prev, vec_smart, vec_dorfman  = [], [], []
prev_list = list(np.linspace(0.05, 0.25, 23)) #Evaluate on prevalences from 5% to 25%
#prev_list =[-1] #Evaluate on original prevalence
for prev in prev_list:
    prevalence, efficiency, random_eff = main(args, prev)
    vec_prev.append(prevalence)
    vec_smart.append(efficiency)
    vec_dorfman.append(random_eff)

h2o.cluster().shutdown()

results_path =os.path.join(output_path, "results.pickle")
pickle_out = open(results_path, "wb")
pickle.dump([vec_prev, vec_dorfman, vec_smart], pickle_out)
pickle_out.close()

if args.savegraph:
    plot_results(results_path,output_path)

