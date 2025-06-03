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

## Installation & Usage

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.0
- Encord SDK (optional for annotation export)
- Azure ML SDK (for cloud experiments)

### Azure ML Pipeline
The project has been successfully deployed for experimentation on Azure ML. 

#### Codebase Explanation:
- 'dataset.py' defines the dataset structure and links to the processed imagery folder within the Azure storage container. Uses the coco annotation url to extract imagery metadata before setting targets for bounding boxes, labels and other key parameters
- 'model.py' defines the function 'get_model' function which creates and returns the model architecture chosen
- 'train.py' completes the tranining loop based on input training parameter arguments and logs metrics using MetricLogger and MLFlow. Example:
```
epochs: 100
batch_size: 4
lr: 5e-05
weight_decay: 5e-05
step_size: 5
gamma: 0.6
```
- 'Training Script.ipynb' authenticates & obtains the Azure ML workspace and submits the training job using 'train.py'.
- Other code files there are helper functions mostly obtained from Pytorch libraries as there were some difficulties in importing the dependencies.

#### Usage:
Below are the steps taken to modify and train the model:
1. Start compute within Azure ML workspace
2. In 'dataset.py' from line 90, class labels can be modified to binary classification by enabling the binary class mapping code from line 95. Otherwise the default is to utilise the 5 classes in the data ontology (+ the background class = 6)
3. Authenticate and activate workspace by running the first cell of 'Training Script.ipynb'
4. Generate the job command using the second cell of 'Training Script.ipynb'. In this cell, modify the name of the model architecture used (RetinaNet or FasterRCNN), the number of classes if using binary classification, the training parameters and link the job correclty to the running compute started in step 1.
5. Run the job in the next cell of 'Training Script.ipynb' using 'ml_client.create_or_update(job)'.
6. As the training uses MetricLogger to track model learning metrics - the visualisations can be seen when viewing the running job in Azure under the 'Metrics' tab. F1, Precision, Recall, Loss and Learning Rate are tracked as the model is trained.

## Future Work
Suggested next steps for improving upon the project include:
- Further dataset expansion for generalisation.
- Model export for edge device inference and integration with StevTech's drone dock and drone hub.
- Initial trial deployment on test farms with StevTech.

## Contributors
- Shahd Sumrain
- Madhav Diwan
- Jacob Ryan
- Sanka Kotinkaduwa

## Acknowledgements
We would like to extend our greatest appreciation to our DMMS mentors David Chambers and Katia Bourahmoune throughout this project - their guidance and knowledge were invaluable. We are also extremely grateful to StevTech for their consistent support and encouragement as we worked on this project. A special thanks to Tuyen Ngyuen - our primary StevTech contact and lead for the computer vision project.


 

