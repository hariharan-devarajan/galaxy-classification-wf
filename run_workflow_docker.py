#!/usr/bin/env python3
# coding: utf-8
import glob 
import os
import numpy as np
from Pegasus.api import *
from pathlib import Path
import logging

# number of workers for the prepreocssing jobs
NUM_WORKERS = 5


logging.basicConfig(level=logging.DEBUG)
props = Properties()
props["pegasus.mode"] = "development"
props.write()

### ADD INPUT FILES TO REPILCA CATALOG
#-------------------------------------------------------------------------------------
rc = ReplicaCatalog()


# list of input file objects
all_images_paths = glob.glob("10_percent_data/*")
input_images = []

for image_path in all_images_paths:
    image_file = image_path.split("/")[-1]
    image_file = File(image_file)
    input_images.append(image_file)
    rc.add_replica("local", image_file,  os.path.join(os.getcwd(), image_path))

        
rc.write()


## ADD TRANSFORMATION SCRIPTS
#-------------------------------------------------------------------------------------
tc = TransformationCatalog()


# define container for the jobs
galaxy_container = Container(
            'crisis_container',
            Container.DOCKER,
            image = "docker://patkraw/galaxy-wf:latest",
            arguments="--runtime=nvidia --shm-size=1gb"
).add_env(TORCH_HOME="/tmp")

# Script that resizes all the images
preprocess_images = Transformation(
                        "preprocess_images",site = "local",
                        pfn = os.path.join(os.getcwd(), "bin/preprocess_resize.py"), 
                        is_stageable = True,container = galaxy_container
                    )
# Script that augments the files of class 2 and 3


# Script that does HPO 

# Script that trains model

# Script thast  
tc.add_containers(galaxy_container)
tc.add_transformations(preprocess_images)
tc.write()
#-------------------------------------------------------------------------------------

## ADD FILE OBJECTS CREATED IN THE WORKFLOW




## CREATE WORKFLOW
wf = Workflow('Galaxy-Classification-Workflow')


def split_preprocess_jobs(preprocess_images_job, input_images, postfix):
    
    resized_images = []
    
    for i in range(len(input_images)):
        curr = i % len(preprocess_images_job)
        preprocess_images_job[curr].add_inputs(input_images[i])
        out_file = File(str(input_images[i]).split(".")[0] + postfix + ".jpg")
        preprocess_images_job[curr].add_outputs(out_file)
        resized_images.append(out_file)
        
    return resized_images


## CREATE JOBS
#-------------------------------------------------------------------------------------
job_preprocess_images = [Job(preprocess_images) for i in range(NUM_WORKERS)]
resized_images = split_preprocess_jobs(job_preprocess_images, input_images, "_proc")

## ADD JOBS TO THE WORKFLOW
wf.add_jobs(*job_preprocess_images)



# EXECUTE THE WORKFLOW
#-------------------------------------------------------------------------------------
try:
    wf.plan(submit=True)
    wf.wait()
    wf.statistics()
except PegasusClientError as e:
    print(e.output)



## VISAULIZE THE WORKFLOW's DAG
graph_filename = "galaxy-wf.dot"
wf.graph(include_files=True, no_simplify=True, label="xform-id", output = graph_filename)



