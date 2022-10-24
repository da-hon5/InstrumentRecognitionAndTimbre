# Interactive T-SNE Visualization
This app was created using C++ and OpenFrameworks. It is inspired by [https://ml4a.github.io/guides/AudioTSNEViewer/](https://ml4a.github.io/guides/AudioTSNEViewer/).

## 1) Create json-file with tSNE-audio.py
To create the json-file containing the file paths to the individual audio clips and their t-SNE embedding run:  
`python tSNE-audio.py --input_dir path/to/input/directory --output_file path/to/output/json`  
For more infos go to: [https://github.com/ml4a/ml4a-ofx/tree/master/apps/AudioTSNEViewer](https://github.com/ml4a/ml4a-ofx/tree/master/apps/AudioTSNEViewer)  
Make sure you do not move the audio clips to another location after doing the analysis, because the paths are hardcoded into the JSON file.

## 2) Run the viewer application
If you are building the application from source, just make sure the variable path is set to point to the JSON file.  
To compile the app:  
`cd ... /instrument-recognition/src/t-sne/AudioTSNEViewer`  
`make`  
`cd bin`  
`./AudioTSNEViewer`  

