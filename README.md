# MVU_Code_Gen

This code generator expects a pure convolutional model without any residual blocks. [Residual disstilation](https://proceedings.neurips.cc//paper/2020/file/657b96f0592803e25a4f07166fff289a-Paper.pdf) 
is one method to remove residual connections from the the model. in ./models directory, you can find some sample models that are already trained to work without any
residual connections at inference.
