### What are Ladder and NOM?
Ladder and NOM are two complementary frameworks designed to simplify neural network training and analysis, making it more accessible to both beginners and experienced researchers. Together, they form a powerful ecosystem that supports model building, visualization, and training in an intuitive and user-friendly manner.  

**Ladder: A Graphical Interface for Neural Network Development**  
Ladder is a graphical user interface (GUI) designed for building, managing, and visualizing neural network models in an intuitive way. Instead of writing complex code, users can construct models by dragging and connecting various components in a visual workspace. This approach makes Ladder an excellent tool for:

  * Beginners: Provides an easy entry point into neural network development without requiring extensive programming knowledge.
  * Research projects: Facilitates experimentation with different architectures and configurations in a structured visual format.
  * Data analysis: Helps in understanding model structures and performance through clear visual representations.
  * Educational purposes: Offers an interactive way for students and instructors to learn and teach deep learning concepts.

**NOM: A Python-Based Neural Network Library**  
NOM (Neural Object Model) is a Python library built on top of TensorFlow, designed to streamline neural network development by providing a simplified, easy-to-read API. Unlike raw TensorFlow, which requires extensive coding knowledge and complex configurations, NOM abstracts many low-level details, allowing users to focus on model design and experimentation.
Key features of NOM include:

  * Configuration-Based Modeling: Users can define models using structured configurations instead of writing extensive code.
  * Seamless Integration with Ladder: NOM interprets the visual model built in Ladder and converts it into TensorFlow-compatible code.
  * Simplified API for Training and Inference: Users can easily collect data, build model architectures, configure hyperparameters, and perform training and predictions without dealing with TensorFlow’s complexity.
  * Framework Independence: While NOM is currently built on TensorFlow, its modular structure allows adaptation to other deep learning libraries, enabling performance comparisons and alternative implementations.

**How Ladder and NOM Work Together**  
When used together, Ladder and NOM create a seamless workflow for neural network development:

1. Model Creation in Ladder: Users visually construct a neural network model using Ladder’s drag-and-drop interface.
2. Data Conversion by NOM: NOM interprets the graphical model and translates it into a structured, TensorFlow-compatible format.
3. Training and Execution in NOM: The converted model is trained using NOM’s streamlined API, leveraging TensorFlow’s powerful deep learning capabilities.
4. Results Visualization in Ladder: Once training is complete, users can view model performance, loss curves, and predictions through Ladder’s visual interface.

By combining Ladder’s graphical capabilities with NOM’s streamlined Python API, this framework reduces the complexity of neural network development, making deep learning more accessible to a wider audience.


### Start Using Ladder
Users can access Ladder to begin their deep learning developments for big data analytics. In the first step, users can create a "project" and build it using a graphical user interface. Projects can be saved as .json files for later sharing or editing. After building a model, users can download a Python file based on NOM. They can either debug the file on an IDE or just click on it to execute in Python. When the Python program finishes executing, users can put the results folder back into Ladder. The tool will then display the training log. As a result, users can edit and view deep learning in a centralized environment. They can build without any Python code except they need to execute the program in Python.

**Step 1: Creating a Project**  
The first step in using Ladder is creating a Project. A project serves as a workspace where you can build and manage your deep learning models. Ladder’s graphical user interface (GUI) allows you to design neural networks visually, eliminating the need for extensive coding.

  * Creating a New Project: Start by defining the purpose of your model—whether for image classification, natural language processing, or table data analyze.
  * Building the Model Graphically: Use drag-and-drop tools to add layers, configure connections, and set hyperparameters. The visual workflow ensures clarity and flexibility.
  * Saving Your Work: Projects are saved as .json files, making it easy to resume, share, or modify models later. These JSON files store all model configurations, ensuring consistency across different development stages.

**Step 2: Exporting and Running the Model**  
Once your model is built, you can export it as a Python script based on NOM (Neural Object Model), the Python library that powers Ladder’s backend processing.

  * Downloading the Python Script: Click the "Train Now" button to generate a Python file. This file contains all necessary configurations to run your model in a Python environment.

  Before running the script, make sure your environment is correctly set up. The required dependencies are shown in [Tutorials/Requirments&Setup](https://cssa.cc.ncku.edu.tw/ladder/get_started/requirements/) page. Including: Python 3.9, NumPy, scikit-learn, TensorFlow.

  if the environment is correctly set up, all you need to do is execute the script, the file contains all necessary configurations to run your model, there is no need to modify the code.
  You can either:
  1. Run Locally – If you prefer to execute the script on your own computer, ensure that your Python environment meets the requirements.
  2. Run on Cloud Computing Resources – You can upload the script to Google Colab or other cloud-based GPUs (e.g., AWS, Google Cloud, or Kaggle Notebooks), and execute it in a cloud environment, reducing the need for local computation.

**Step 3: Analyzing the Training Results in Ladder**  
After the model completes training, it generates a Results Folder, containing logs, model checkpoints, and performance metrics. 

Users can upload the results folder back into Ladder. Visualizing Training Logs: Ladder automatically processes the results and displays:
* Loss Curves – Shows how the model improves over time.
* Accuracy Metrics – Helps evaluate model performance.
* Trace Logs – Provides insights into execution steps and errors.

**Need Help Getting Started?**  
If you’re new to Ladder, we provide an easy-to-follow tutorial to guide you through the essential steps:

Using renowned image datasets and tabel data as examples.  
Creating and editing model in Ladder.  
Training your model and exporting it as a Python script.  
Analyzing results and logs.  
Visit our [Tutorial](https://cssa.cc.ncku.edu.tw/ladder/get_started/) Page to learn how to build and train your first model with Ladder!


### Supported Data in Ladder
Ladder supports two primary types of datasets for training: data tables and image datasets. Users can either upload their own datasets or choose from preloaded well-known datasets such as MNIST and CIFAR-10, which are available for download from the [Explore Data](https://cssa.cc.ncku.edu.tw/ladder/get_started/explore/) page. These preloaded datasets come with predefined labels, making it easier to identify the "Input" and "Target" features automatically.

**1. Data Tables (CSV Format)**  
  For data tables, Ladder currently only supports CSV files. These files are processed locally on the user’s device, ensuring privacy as no data is transmitted over the internet. CSV files allow users to work with tabular datasets, which can include numerical, categorical, or mixed data types.

  Users have multiple options for importing data table into Ladder:  
  **Uploading a CSV File:** Select Upload a CSV file from your device.  
  **Pasting Data Directly:** If you have data copied from a spreadsheet application (e.g., Excel, Google Sheets), simply paste it into Ladder using Ctrl + V.
  **Skip file selection or pasting:**  User can skip file selection or pasting in the first place, but must manually specify the "Column Count" and "File Location". And without imported data, no column preview will be available.

  File Size and Performance Considerations:  
  While Ladder does not enforce a strict file size limit, it is recommended to use CSV files under 1GB for optimal performance. Larger datasets may lead to increased processing time and memory usage, especially on lower-end devices.

  Column Configuration and Data Embedding:  
  If a CSV file is uploaded, Ladder will automatically detect the column structure and provide a column preview.
  Data Embedding: Ladder stores an internal copy of the uploaded or pasted dataset, allowing users to edit and preview column structures conveniently.
  If a File Location is specified, the model will reference the external file for training and testing. Otherwise, Ladder will default to using the embedded data.

**2. Image Datasets (Preloaded and Custom Uploads)**  
  For deep learning applications, Ladder supports image datasets for tasks such as image classification.

  1. Preloaded Datasets:  
  Ladder provides easy access to renowned image classification datasets, including:
  MNIST, CIFAR-10. These datasets can be downloaded directly from the [Explore Data](https://cssa.cc.ncku.edu.tw/ladder/get_started/explore/) page and come with predefined input (images) and target (labels), eliminating the need for manual feature selection.

  2. Using Custom Image Datasets:  
  Ladder allows users to upload and configure their own image datasets for training deep learning models. Users can choose between two types of image-based tasks:  
    * Image to Classification: Assigns labels to images for tasks like object recognition.  
    * Image to Image: Used for tasks like image translation or enhancement.

  Setting Up the Training Data:  
  To begin, users must specify the Folder Location where their training images are stored. Supported image formats include .jpg, .jpeg, .png, .gif, and .bmp. Users can manually enter the folder path or browse their device to select the correct location.  

  For Image to Classification task, users need to upload a CSV file containing image file names and their corresponding labels.


### Configure Test Data
Ladder provides flexible options for defining test data, allowing users to either specify a separate test dataset or adjust the test set proportion, depending on the dataset type. This ensures proper validation while preventing data leakage.

**1. Test Data Handling for Preloaded Image Datasets**  
For renowned image datasets such as MNIST or CIFAR-10, Ladder automatically uses the predefined training and test splits. Users cannot specify an independent test dataset or adjust the test set proportion for these datasets.

**2. Configuring Test Data for Custom Image Datasets**  
For custom image datasets, users have full control over test data configuration:  
  * Users can specify an independent test dataset by providing a separate test image folder and a corresponding test label CSV file.  
  * Alternatively, if a test dataset is not provided, Ladder will automatically split test dataset from  the original dataset. Users can set the test proportion in the Properties Panel of the data source node in the model graph.  

**3. Test Data Handling for Data Table Datasets**  
For structured data (data tables):  
* Users cannot specify an independent test dataset.
* However, they can adjust the test proportion from the original dataset in the Properties Panel of the data source node in the model graph.


### Configuring Data Sources and Image Preprocessing in Ladder
1. Modifying CSV Data Sources:  
In Ladder, users can modify the file location of a CSV data source by selecting the data source node in the model graph or right-clicking the node and choosing "Properties" from the context menu. The "Folder Location" can then be updated in the right sidebar. However, Ladder currently does not support re-pasting a data table once a project is opened; users need to create a new project instead. 
2. Image Preprocessing in Ladder:  
For image datasets, Ladder has not yet implemented manual data preprocessing options. When training CIFAR-10, automatic data augmentation techniques such as random flipping, brightness adjustment, contrast adjustment, and cropping are applied, with the crop size adjustable in the "Input" node properties. MNIST datasets do not currently undergo any data augmentation. Users should stay updated on future developments regarding image data preprocessing. Additionally, Ladder supports circular data, where the maximum and minimum values are contextually the same, such as color hue and angular values.


### Adding and Removing Hidden Layers
Ladder provides users with multiple intuitive ways to add and remove hidden layers in a neural network model. These options allow for flexible model design while ensuring mathematical validity.  

Ladder offers multiple ways to create a new hidden layer. 
1. Users can click the "+" button at the bottom of the model graph to add a layer at the end of the model. 
2. Users can right-click an existing layer or data preprocessing node and choose "Insert After" to place a hidden layer before its subsequent connections. 
3. Another option is selecting "Create a Layer" to append a new layer, which may create a branch if the existing layer is already connected to another one. 
4. Users can also drag a desired model layer from the left-side toolbar onto the "+" button at the intended location. 

To remove a hidden layer, simply right-click the layer and select "delete".

However, inserting or removing a hidden layer may sometimes be restricted due to mathematical conflicts. Since Ladder will automatically reconnect after amending a middle layer, if the changes bring with mathematical conflicts, say the later layer may have specific requirements on the incoming layer matrix shape, this will block the action of inserting/removing the layer.
Please check with the shape restrictions of later layers. You may delete all layers afterwards to ensure a layer can be inserted just after the existing layer.


### Project Saving and Load Previous Edited Model
Ladder allows users to save their neural network models and training configurations for future use. This ensures that work is not lost and enables easy collaboration by sharing saved models with others. The saved project files follow the NOM (Neural Object Model) JSON format, which retains all relevant model settings, including architecture, hyperparameters, and training configurations.

To save a project, press the "Project" button at the upper-right corner of the interface, and then press "Save Project" button to save the project as an NOM-defined JSON file.

To resume work on a saved project, go to Ladder start screen, click "Open a Neural Object Model", then everything will be resumed.

Note: Project will only save in NOM format, and can only be loaded and used in Ladder platform.


### Adjusting Train Settings and Training Models
Ladder provides a flexible interface for configuring various training settings and hyperparameters, allowing users to fine-tune their models for optimal performance. These settings can be adjusted through the right-side toolbox, ensuring that users have full control over how their model is trained.

**Configurable Training Settings:**  
Users can modify the following parameters before starting the training process:

* Number of Epochs: Defines how many times the model will iterate over the entire dataset.
* Cross-Validation Type: Allows users to choose between different validation techniques, such as k-fold cross-validation or leave-one-out validation.
* Batch Size: Determines the number of samples processed before the model updates its weights.
* Data Shuffling: Enables or disables shuffling of training data before each epoch.
* Repeated Runs Count: Sets the number of times the entire training process should be repeated. This is useful for running multiple trials with the same configuration to observe variations in performance due to factors like data shuffling or weight initialization.
* Tracking Frequency: Specifies how often performance metrics and loss values are recorded during training.
* Log Frequency: Controls how frequently training logs are saved for monitoring progress.
* Save Frequency: Defines how often model checkpoints are saved during training.
* Test Frequency: Determines how frequently the model is evaluated on the test set.
* Weight Frequency: Specifies how often the model’s weight parameters are recorded.
* Trace Count & Trace Frequency: Configure the number of recorded training traces and their logging frequency.

**Saving Projects and Train Scripts:**  
Once the model is configured, users can proceed with one of the following options:

1. Saving the Project:  
    1. Click the "Project" button in the upper-right corner.
    2. Choose to save the project in either:
        * NOM format (project.json) – click "save project", allows users to reload and edit the project later or share with others in Ladder.
        * Python Script (train.py) – click "save code", will generate a script that can train the model independently. Click the "Train Now" button will also generate a train.py script, user can follow the instructions afterwards and executed locally.

**Executing the Training Script:**
Ladder does not provide cloud computing resource, user must run the training script outside Ladder, either run locally or on a cloud computing resource. And open the result folder back in Ladder.

Before running the script, make sure your environment is correctly set up. The required dependencies are shown in [Tutorials/Requirments&Setup](https://cssa.cc.ncku.edu.tw/ladder/get_started/requirements/) page. Including: Python 3.9, NumPy, scikit-learn, TensorFlow.

  if the environment is correctly set up, all you need to do is execute the script, the file contains all necessary configurations to run your model, there is no need to modify the code.
  You can either:
  1. Run Locally – If you prefer to execute the script on your own computer, ensure that your Python environment meets the requirements.
  2. Run on Cloud Computing Resources – You can upload the script to Google Colab or other cloud-based GPUs (e.g., AWS, Google Cloud, or Kaggle Notebooks), and execute it in a cloud environment, reducing the need for local computation.

**Reviewing Training Results:**
Once training is complete, the script will generate a results folder containing the training settings configured by the user. Open this folder in Ladder to review the training outcomes.

By following these steps, users can efficiently configure, train, and analyze their models using Ladder.


### Train Logs and Results
After completing the training process, Ladder provides tools to help users analyze and interpret the training performance through detailed logs and visualizations. Users can upload their results folder to Ladder to access the train log and trace log, which contain important training metrics.

**Accessing Training Logs:**  
To review the results, click the "Open a Results Folder" button. This will load the training logs based on the configuration settings applied before training. The logs provide insights into model performance, helping users diagnose issues, track progress, and refine their models accordingly.

**Understanding the Train Log:**  
The train log presents a graphical representation of key training metrics, including loss values and learning rates. It helps users evaluate the model's learning progress.

* X-Axis (Global Step):
  * Represents the total number of optimization steps taken during training.
  * Each step corresponds to one batch of data processed by the model.
  * A higher number indicates a longer training duration.
* Y-Axis (Loss Value, Green Line):
  * Displays the average loss at each training step.
  * Loss measures the difference between the model’s predictions and the actual labels.
  * A decreasing loss curve suggests the model is learning effectively.
* Y-Axis (Learning Rate, Blue Line):
  * Shows how the optimizer adjusts the learning rate over time.
  * Learning rate dictates the size of each update step toward optimizing model weights.
  * It usually starts higher and decreases according to a predefined schedule, preventing overshooting while ensuring steady progress.

**Understanding the Trace Log:**  
The trace log provides a more granular view of model training by capturing training results at specified intervals based on the trace frequency setting.

* Displays performance metrics such as accuracy, loss, or validation results at different checkpoints.
* Helps identify trends and anomalies during training.
* Useful for debugging and determining the optimal stopping point for training.

By analyzing both the train log and trace log, users can gain valuable insights into their model's behavior, fine-tune hyperparameters, and make informed decisions about further training or adjustments.


### Supported Layers on Ladder
**Below is a list of layer types available on Ladder:**  

1. **MobileNet Family** - MobileNet V1, V2, V3 (small/large), optimized for mobile and edge devices.  
2. **Basic Layers** - FCL (fully connected), Conv/DeConv (2D), Conv_1d/Deconv_1d (1D), Pooling (Max/Average).  
3. **Recurrent Layers** - RNN, GRU, LSTM (handle long sequences), BiRNN, BiGRU, BiLSTM (bidirectional versions).  
4. **Normalization & Activation** - BN (Batch Norm for stability), Softmax (logits to probabilities).  
5. **Advanced Convolutions** - Conv1d/2d/3d, D_conv2d (efficient depthwise conv, used in MobileNets).  
6. **Pooling & Flattening** - Max/Avg Pooling (downsampling), Flatten (prepares for dense layers).  
7. **Manipulation Layers** - Reshape, Concat (tensor merging), Zeros/Ones (tensor initialization).  
8. **Math Operations** - Add, Multiply, Divide, Pow, Sqrt, Exp, Mean, Square.  
9. **Dropout & Noise** - Dropout (prevents overfitting), Noise (adds robustness).  
10. **External Layers** - TF Hub (imports pre-trained models).  
11. **Special Layers** - Collector (A custom layer for collecting output at specific points in the model), Task (A layer specific to Ladder for integrating task-based modules.).


### Special Layers
Ladder allows users to create bypassing layers, such as those in ResNet, by enabling layer connection mode. This can be done by hovering over a layer and clicking the directional arrow button or by right-clicking a layer and selecting "Attach on Layer" or "Connect to Layer." Once in connection mode, users can click the arrows next to layers to modify their connections. When a layer receives inputs from multiple layers, its Properties Pane includes an "Incoming Connections" section that defines how the incoming matrices are processed. Ladder’s high-level layers support auto-connections, allowing users to concatenate, sum, multiply, or blend layers with different matrix sizes. If incoming matrices have varying dimensions, a learning layer is automatically applied to adjust them to match the core incoming layer, which is typically the layer with the lowest dimension. Additionally, convolutional layers can be used on non-image data. Ladder will automatically reshape incoming data into a 4D format (Batch/Height/Width/Channel) if necessary, making convolutional layers potentially useful for serial data. Finally, Ladder includes a special high-level layer called the Collector, which automatically receives inputs from multiple layers without performing an explicit linear transformation like a fully connected layer. However, it still offers configurations such as activation functions and output reshaping.


### Supported Activation Functions on Ladder
**Below is a list of available activation functions on Ladder:**  

1. **ReLU** - Outputs input if positive; otherwise, zero. Common in deep learning due to simplicity and effectiveness.  
2. **Sigmoid** - Compresses input to [0,1], useful for binary classification.  
3. **Tanh** - Scales input to [-1,1], aiding faster learning; used in RNNs.  
4. **Hard Sigmoid** - A faster, piecewise linear approximation of Sigmoid, optimizing performance.  
5. **Linear** - Identity function, used in regression output layers.  
6. **ReLU6** - A ReLU variant capping output at 6, useful for mobile applications.  
7. **SELU** - Self-normalizing, maintaining mean and variance across layers.  
8. **ELU** - Allows negative values with an exponential term for noise robustness.  
9. **Softplus** - A smooth ReLU approximation with a non-zero gradient.  
10. **Softsign** - Similar to Tanh but with a slower gradient change.  
11. **CReLU** - Applies ReLU to input and its negation, doubling activations.


### Developers and Development History
We are the Creative System and Software Applications Laboratory from National Cheng Kung University, Taiwan. Our lab specializes in modern software design, algorithm development, and applied technologies. Ladder is our first open-source project, and we plan to continue enhancing its features and user experience, with a focus on educational and industry applications.
The concept was proposed in early 2018, and the related results were published in I-SPAN 2018. 

The demand for artificial intelligence tools has surged in recent years, particularly among small and medium-sized enterprises (SMEs) looking to leverage neural network technology for deep data analysis. However, many companies lack the personnel who possess both domain knowledge and coding skills.

Ladder and NOM address this gap by offering:
* **Accessibility**: An easy-to-use interface for model creation, without the need for coding expertise.
* **Scalability**: Users can begin with visual model editing and gradually transition to code-level customization.
* **Education and SME Focus**: These tools are designed to promote AI knowledge dissemination, making neural network technology more accessible to students and small businesses.