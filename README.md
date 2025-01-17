This is the code accompanying paper "Data driven retinotopic image deblurring in diffuse optical tomography"
You will need NITFASTer toolbox, NeuroDOT and PyTorch to run the code

Training data was generated with "GenerateTrainingData.m"

Testing data was generated with "CreateSpinningDataWithNoise.m"

The UNET model is trained with "UNET.py"

To process data with the UNET model run the script "UNET Eval.py", and to visualise use "LookAtVisualMapm.m" (preferable) or "Visualisation.py"

"ClassicalPiplineFinal.m" was used to process the experimental data supplied with neuroDOT

Also included are,
- E.mat: relevant extinction coefficients for 750nm and 850nm
- Good_Vox.mat: neuroDOT Good_Vox, specifying the indexes of voxels significant values of the Jacobian
- mask.mat: A mask of the gray matter that was dilated so the network only computes loss on the relevant volume
- mesh: the high density mesh used for simulations and calculating the inverse jacobian

For further questions and enquiries please contact:
- Joe Evans: j.o.evans@bham.ac.uk
- Prof. Hamid Dehghani: h.dehghani@bham.ac.uk
