#!/usr/bin/env python3

import glob 
import os
import numpy as np
from Pegasus.api import *
from pathlib import Path
import logging
import pickle
import time


logging.basicConfig(level=logging.DEBUG)
props = Properties()
props["pegasus.mode"] = "development"
#props["dagman.retry"] = "3"
#props["pegasus.transfer.arguments"] = "-m 1"
props.write()



def get_arguments():
    
    parser = argparse.ArgumentParser(description="Galaxy Classification")   
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--data_path', type=str, default='10_percent_data/',help='path to dataset ')
    parser.add_argument('--epochs', type=int,default=1, help = "number of training epochs")  
    parser.add_argument('--trials', type=int,default=1, help = "number of trials") 
    parser.add_argument('--num_workers', type=int, default= 5, help = "number of workers")
    parser.add_argument('--num_class_2', type=int, default= 3, help = "number of augmented class 2 files")
    parser.add_argument('--num_class_3', type=int, default= 4, help = "number of augmented class 3 files")
    parser.add_argument('--maxwalltime', type=int, default= 10, help = "maxwalltime")
    args = parser.parse_args()
    
    return args


def create_pkl(name):
    pkl_filename = name
    file = open(pkl_filename, 'ab')
    pickle.dump("", file, pickle.HIGHEST_PROTOCOL)
    return pkl_filename




def split_preprocess_jobs(preprocess_images_job, input_images, postfix):
    
    resized_images = []
    
    for i in range(len(input_images)):
        curr = i % len(preprocess_images_job)
        preprocess_images_job[curr].add_inputs(input_images[i])
        out_file = File(str(input_images[i]).split(".")[0] + postfix + ".jpg")
        preprocess_images_job[curr].add_outputs(out_file)
        resized_images.append(out_file)
        
    return resized_images



def add_augmented_images(class_str, num, start_num):
    augmented_files = []
    for i in range(num):
        augmented_files.append(File("train_" + class_str + "_" + str(start_num) + "_proc.jpg"))
        start_num +=1
    return augmented_files


def create_files_hpo(input_files):
    files = []
    for file in input_files:
        name = File(file.split("/")[-1].split(".")[0] + "_proc.jpg")
        files.append(name)
    return files

def run_workflow():


	### ADD INPUT FILES TO REPILCA CATALOG
	#-------------------------------------------------------------------------------------------------------
	rc = ReplicaCatalog()

	# list of input file objects
	all_images_paths    = glob.glob( REL_PATH + '*.jpg')
	train_files_class_2 = glob.glob(REL_PATH + 'train_class_2*.jpg')
	train_files_class_3 = glob.glob(REL_PATH + 'train_class_3*.jpg')

	all_train_files     = glob.glob(REL_PATH + 'train_class_*.jpg')
	all_val_files       = glob.glob(REL_PATH + 'val_class_*.jpg')
	all_test_files      = glob.glob(REL_PATH + 'test_class_*.jpg')

	input_images = []

	for image_path in all_images_paths:
	    image_file = image_path.split("/")[-1]
	    image_file = File(image_file)
	    input_images.append(image_file)
	    rc.add_replica("local", image_file,  os.path.join(os.getcwd(), image_path))

	# ADDITIONAL PYTHON SCRIPS NEEDED BY TUNE_MODEL
	#-------------------------------------------------------------------------------------------------------
	data_loader_fn = "data_loader.py"
	data_loader_file = File(data_loader_fn )
	rc.add_replica("local", data_loader_fn, os.path.join(os.getcwd(), "bin/" + data_loader_fn ))

	model_selction_fn = "model_selection.py"
	model_selction_file = File(model_selction_fn )
	rc.add_replica("local", model_selction_fn, os.path.join(os.getcwd(),"bin/" + model_selction_fn ))


	# FILES FOR vgg16_hpo.py VGG 16
	#--------------------------------------------------------------------------------------------------------
	vgg16_pkl = create_pkl("hpo_galaxy_vgg16.pkl")
	vgg16_pkl_file = File(vgg16_pkl)
	rc.add_replica("local", vgg16_pkl, os.path.join(os.getcwd(), vgg16_pkl))	

	# FILES FOR train_model.py 
	#--------------------------------------------------------------------------------------------------------
	checkpoint_vgg16_pkl = create_pkl("checkpoint_vgg16.pkl")
	checkpoint_vgg16_pkl_file = File(checkpoint_vgg16_pkl)
	rc.add_replica("local",checkpoint_vgg16_pkl, os.path.join(os.getcwd(), checkpoint_vgg16_pkl))

	rc.write()

	# TRANSFORMATION CATALOG
	#---------------------------------------------------------------------------------------------------------
	tc = TransformationCatalog()

	# Data preprocessing part 1: image resize
	preprocess_images = Transformation("preprocess_images",site="local",
	                                pfn = str(Path(".").parent.resolve() / "bin/preprocess_resize.py"), 
	                                is_stageable= True)

	# Data preprocessing part 2: image augmentation
	augment_images = Transformation("augment_images",site="local",
	                                pfn = str(Path(".").parent.resolve() / "bin/preprocess_augment.py"), 
	                                is_stageable= True)

	# HPO: main script
	vgg16_hpo      = Transformation("vgg16_hpo",site="local",
	                                pfn = str(Path(".").parent.resolve() / "bin/vgg16_hpo.py"), 
	                                is_stageable= True)

	# Train Model
	train_model     = Transformation("train_model",site="local",
	                                pfn = str(Path(".").parent.resolve() / "bin/train_model.py"), 
	                                is_stageable= True)

	# Eval Model
	eval_model     = Transformation("eval_model",site="local",
	                                pfn = str(Path(".").parent.resolve() / "bin/eval_model.py"), 
	                                is_stageable= True)

	tc.add_transformations(preprocess_images,augment_images, vgg16_hpo, train_model,eval_model )
	tc.write()

	## CREATE WORKFLOW
	#---------------------------------------------------------------------------------------------------------
	wf = Workflow('Galaxy-Classification-Workflow')

	job_preprocess_images = [Job(preprocess_images) for i in range(NUM_WORKERS)]
	resized_images = split_preprocess_jobs(job_preprocess_images, input_images, "_proc")

	input_aug_class_2 = [ File(file.split("/")[-1].split(".")[0] + "_proc.jpg") for file in train_files_class_2 ]
	output_aug_class_2 = add_augmented_images("class_2", NUM_CLASS_2, 4000)

	input_aug_class_3 = [ File(file.split("/")[-1].split(".")[0] + "_proc.jpg") for file in train_files_class_3 ]
	output_aug_class_3 = add_augmented_images("class_3", NUM_CLASS_3, 4000)


	job_augment_class_2 = Job(augment_images)\
	                    .add_args("--class_str class_2 --num {}".format(NUM_CLASS_2))\
	                    .add_inputs(*input_aug_class_2)\
	                    .add_outputs(*output_aug_class_2)

	job_augment_class_3 = Job(augment_images)\
	                    .add_args("--class_str class_3 --num {}".format(NUM_CLASS_3))\
	                    .add_inputs(*input_aug_class_3)\
	                    .add_outputs(*output_aug_class_3)


	all_train_files = glob.glob(REL_PATH + 'train_class_*.jpg')
	all_val_files   = glob.glob(REL_PATH + 'val_class_*.jpg')
	all_test_files  = glob.glob(REL_PATH + 'test_class_*.jpg')

	input_hpo_train = create_files_hpo(all_train_files)
	input_hpo_val   = create_files_hpo(all_val_files)
	input_hpo_test  = create_files_hpo(all_test_files)


	best_params_file = File("best_vgg16_hpo_params.txt")

	# Job HPO
	job_vgg16_hpo = Job(vgg16_hpo)\
	                    .add_args("--trials {} --epochs {} --batch_size {}".format(TRIALS, EPOCHS, BATCH_SIZE))\
	                    .add_inputs(*output_aug_class_3, *output_aug_class_2, \
	                                *input_hpo_train, *input_hpo_val,data_loader_file, model_selction_file)\
	                    .add_checkpoint(vgg16_pkl_file, stage_out=True)\
	                    .add_outputs(best_params_file)\
	                    .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=MAXTIMEWALL)


	# Job train model
	job_train_model = Job(train_model)\
	                    .add_args("--epochs {} --batch_size {}".format( EPOCHS, BATCH_SIZE))\
	                    .add_inputs(*output_aug_class_3, *output_aug_class_2, best_params_file, \
	                                *input_hpo_train, *input_hpo_val, *input_hpo_test,\
	                                data_loader_file, model_selction_file)\
	                    .add_checkpoint(checkpoint_vgg16_pkl_file , stage_out=True)\
	                    .add_outputs(File("final_vgg16_model.pth"),File("loss_vgg16.png"))\
	                    .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=MAXTIMEWALL)


	# Job eval
	job_eval_moodel = Job(eval_model)\
	                    .add_inputs(*input_hpo_test,data_loader_file,best_params_file,\
	                                model_selction_file,File("final_vgg16_model.pth"))\
	                    .add_outputs(File("final_confusion_matrix_norm.png"),File("exp_results.csv"))


	## ADD JOBS TO THE WORKFLOW
	wf.add_jobs(*job_preprocess_images,job_augment_class_2 ,job_augment_class_3, job_vgg16_hpo,\
	            job_train_model,job_eval_model)  


	# EXECUTE THE WORKFLOW
	#-------------------------------------------------------------------------------------
	try:
	    wf.plan(submit=True)
	    wf.wait()
	    wf.statistics()
	except PegasusClientError as e:
	    print(e.output)   



def main():
    
    start = time.time()
    
    global ARGS
    global BATCH_SIZE
	global SEED
	global DATA_PATH
    global EPOCHS
	global TRIALS
    global NUM_WORKERS
	global NUM_CLASS_2
	global NUM_CLASS_3
	global MAXTIMEWALL
	global REL_PATH

    ARGS        = get_arguments()
    BATCH_SIZE  = ARGS.batch_size
    SEED        = ARGS.seed
    DATA_PATH   = ARGS.data_path
    EPOCHS      = ARGS.epochs
    TRIALS      = ARGS.trials
    NUM_WORKERS = ARGS.num_workers
    NUM_CLASS_2 = ARGS.num_class_2
    NUM_CLASS_3 = ARGS.num_class_3
    MAXTIMEWALL = ARGS.maxwalltime
    REL_PATH = '10_percent_data/'


    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_workflow()
    
    exec_time = time.time() - start

    print('Execution time in seconds: ' + str(exec_time))
    return

if __name__ == "__main__":
    
    main()