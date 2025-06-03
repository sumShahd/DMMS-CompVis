# Computer Vision Weed Detection Model
_A project completed as part of the UTS subject Design in Mechanical and Mechatronic Systems for StevTech Pty Ltd._

## Project Overview
This project explores the development of an AI-driven computer vision model for green-on-brown weed detection in Australian agricultural fields during the fallow phase. The model aims to localise weed presence using UAV imagery and object detection methods to improve chemical efficiency, reduce labour demands, and offer scalable precision agriculture solutions.

### Problem Statement
Weeds during the fallow phase compete with soil nutrients and moisture, directly reducing crop yields.
Current weed management methods rely heavily on blanket herbicide spraying, leading to:
- Unnecessary chemical use and runoff
- Increased farm costs
- Herbicide-resistant weed species
- Labour and scalability issues across large farms

This project proposes a computer vision solution to localise weeds in UAV imagery, offering the opportunity for targeted spraying which will work towards minimising the above issues.

## Data Ontology
The dataset uses a custom object detection-based ontology rather than image-level classification as the detected weeds need to be localised to enable target spraying. Classes are grouped morphologically, focusing on generalisation across farms:

### Weed Classes
- Broadleaf Weeds
- Grassy/Grass-Like Weeds
- Woody Shrub-Like Weeds
- Generalised Green - for weeds with unclear shape/type from imagery.

### Obstacle Class/es
- Farm Infrastructure (buildings, fences, powerlines)
- Vegetation (non-weed trees & plants)
- Bodies of Water (ponds, creeks, swamp)
- Machinery (farm vehicles and equipment)

**NOTE: The above obstacle classes were integrated into a singular obstacle class for better class balance across the dataset.**

### Dataset
Drone imagery were provided by StevTech as 512x512 image tiles of all farm imagery obtained. Annotation and dataset creation were conducted using Encord and annotations followed a JSON-based COCO format. The dataset was split as follows:
- Train: 90%
- Validation: 10%
- Test: Unseen drone imagery

## Model Overview
We initially selected RetinaNet with a ResNet50 backbone as the primary model after a literature survey due to it offering a balance of accuracy, speed, and potential for class weight modification of the loss function. We also trained a model with the FasterRCNN architecture with binary classification (one weed class combing all and a background class combining the obstacle classes) which acted as an initial classifier for the drone image tiles (most of which did not contain any weeds). Additional model training details:

**Framework:** PyTorch
**Machine Learning Platform:** Azure ML 
**Training method:** Fine-tuned on COCO pre-trained weights
**Target metrics:** ≥90% accuracy (F1-score) with 10 FPS inference
**Compute Specifications:** 
- GPU: 1 x NVIDIA Tesla T4
- Virtual machine size: Standard_NC4as_T4_v3 (4 cores, 28 GB RAM, 176 GB disk)

##Contributors
- Shahd Sumrain
- Madhav Diwan
- Jacob Ryan
- Sanka Kotinkaduwa



 

