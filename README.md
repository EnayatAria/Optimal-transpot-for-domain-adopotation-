# Optimal-transpot-for-domain-adopotation-

This is a repository  for applying optimal transport to remote senisng images as a domain adoptation method.


This code is for applying optimal transport as a domain adaptation algorithm to remote sensing images
Four different OT functions is used in this code: Earth moving distance (EMD), Sinkhorn, Group Sparsity, and Laplace
I used the inverse transformation in this code: i.e. target data is mapped into source data.


Input:

   - source image dataset in envi format
   - target image dataset in envi format
   - training samples of the classes of source dataset
    
    
Output:

   - four transformed image sets using  four OT functions in envi format
   - Plot of the point distributions of the source and target data and the transformed target images

A sample plot as the output of the plot  output of the code is uploaded: tf_plot_best.jpeg 
