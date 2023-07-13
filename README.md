# Depixilation
    The goal of this project was to "depixelate" pixelated parts of images.

    Since the test set, on which the evaluation of our models took place, was based on 64x64 grayscale image
    dataset with certain parts pixelated, the following work tailored towards this special case.

    During training pixelation was done randomly on each image and both the pixelated image, as well as a
    mask which signalled where on the image the pixelation took place, was fed into an according architecture.

    This means the model did NOT have to learn to infer where pixelation took place and could focus solely on
    the task of incribing appropiate pixels in the respective fields.

## Project Structure
    The files datasets.py and data_utils.py feature a variety of Code needed for the RandomImagePixelation
    dataset.

    Whereas utils.py provides general utility for the project (plotting, kernel size interpolation,
    serializing test predictions for submission ...).

    Of course architectures.py features all the different architectures I tried during the many tests on
    smaller datasets and main.py is mainly used for initiating the training loop and directly getting a
    qualitative look at the predictions of the model on a small number of hand-picked samples from the
    originally provided dataset.
    Also, this file can be used as a script in conjunction with torchinfo (https://github.com/TylerYep/torchinfo)
    to get a good overview of a certain instanced architecture and also estimate training loads based on batch
    size.

    Models which served more a purpose of trying out different hyperparameters are stored under models/, whereas
    the seriously trained model on a much larger dataset (original data + STL10 unlabeled images) where stored
    under models_serious/.
    
    Under losses/ the according monitored training and evaluation loss can be seen.

## Architecture
    test
        test
