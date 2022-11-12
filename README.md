# HotelBookingCancellation

## Description

### Overview

Main idea of the project, business pain, what was used to solve it and its firsts results

Ex: We wanted to validate if the demand behaviour of our clients, ex: gasoline and alcohol demand, for a specific month is normal, inside some boundries, or not. To do that we used the algorithm XPTO and in our firsts results we reduced the fee for wasted product / or reduced the amount of debt not paid by clients by XX% / in R$ YY, because we denied purchases that were out ou the client usual behaviour

### Motivation

Business pain we wanted to solve. Why has this project been developed?

### Dataflow Diagram

Design a simple descriptive dataflow diagram. You can use [Kedro Viz](https://github.com/kedro-org/kedro-viz), [Mermaid](https://mermaid-js.github.io/mermaid/#/) and [MermaidCLI](https://github.com/mermaid-js/mermaid-cli#transform-a-markdown-file-with-mermaid-diagrams) and [VSCode Mermaid](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid), [PlantUML](https://plantuml.com/) or a custom markdown image under the `docs/assets` folder.

### Pipelines

Enter a simple description of the `Kedro` pipelines. If possible, use the same tools used in the DataFlow section for explaining it.

_Example:_

* **Data Engineering:** Pre-processes the inputs X, Y, Z using moving averages in order to smooth the prediction results producing A, B, C.

* **Data Science:** Predicts the inputs A, B, C values for the post three months from the execution moment using linear regression producing outputs D, E, F. 

### Inputs/Features

This project requires the following data/artifacts:

_Example:_

#### Artifacts

| Artifact | Type    | Notes  |
| -------- | ------- | ------ |
| iris     | Tabular |        |

#### Features

| From     | Feature      | Notes  |
| -------- | ------------ | ------ |
| iris     | Sepal length |        |
| iris     | Sepal width  |        |
| iris     | Petal length |        |
| iris     | Petal width  |        |
| iris     | Species      | Target |

### Outputs

This project generates the following data/artifacts:

_Example:_

| Artifact   | Type    | Notes            |
| --------   | ------- | ---------------- |
| Classifier | Model   |                  |
| Report     | Metrics | Accuracy, Recall |

### Algorithm explanation

Explain the main data science process, what algorithm was chosen, why it was chosen, and how it solves the problem.

#### Data preparation

Describe how you cleaned and processed data for the post steps.

#### Feature Engineering

Describe what criteria did you use to choose the actual features for this model.

#### Optimization

Describe how you trained the chosen algorithm, what metrics you used, and how you validated its result.

### Use Cases

Describe some possible scenarios for using this project outputs.

## Usage

### Installation

In order to run this project, execute the following steps:

1. Clone this repo
2. Go to the project folder using terminal
3. Run `pip install -r src/requirements.txt`

### Execution

For executing the pipelines mentioned before, run the following commands:

1. Execute `kedro <projetaai-plugin> init` for creating the missing local files
2. Run `kedro run --pipeline <pipeline_name>`

#### Notes

If required, add any more information the user should know for using this pipeline. For example, login operations.

## Development

There are some other tools required for changing the project source code. Execute the commands below:

1. Install dev deps with `pip install -r src/requirements-dev.txt`
2. Install test deps with `pip install -r src/requirements-test.txt`
3. Install pre-commit with `pre-commit install`
4. If unit tests were created, run `pytest` before committing to ensure no breaking changes were made.

## Authors

Enter the name and email of the authors in this section using a bullet list:

* Nickolas da Rocha Machado; [nickolasrochamachado@gmail.com](mailto:nickolasrochamachado@gmail.com)

## References

Fill this section with the articles and papers that were relevant to developing this solution.

_Example:_

1. [XOR-Net: An Efficient Computation Pipeline for Binary Neural Network
Inference on Edge Devices](https://cmu-odml.github.io/papers/XOR-Net_An_Efficient_Computation_Pipeline_for_Binary_Neural_Network_Inference_on_Edge_Devices.pdf)
